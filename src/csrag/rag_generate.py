"""Generation for notebook 11 (x3_to_llm.md): answers from the X3 contexts, per LLM and prompt.

The notebook only calls into this module. For every (LLM, prompt) it writes one JSONL file, one
record per question, as the answers come in; the last cell combines the four files into
``rag_output.jsonl``. Nothing here needs Qdrant or the encoder: it reads
``results/<experiment>/x3/contexts.jsonl`` (notebook 10) and the question files, and talks to the
vLLM container.

    items = load_inputs(EXP, RESULTS / "x3" / "contexts.jsonl")      # contexts joined with the questions
    run_combination(items, "qwen", "simple", combo_path(OUT, "qwen", "simple"))
    ...
    combine(OUT, items, ["qwen", "gemma"], ["simple", "robust"], OUT / "rag_output.jsonl")

No cache (the output files are the record; ``data/cache/llm.jsonl`` is committed to git and belongs
to the retrieval-side LLM calls). The prompts are in ``rag_prompts.py`` and are reloaded from that
file at the start of every run, so an edit takes effect without restarting the kernel.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

from . import rag_prompts

# key -> label (used in file names and in the records), the model id vLLM serves, and how many requests
# run at once (= max-num-seqs of the serve file: vllm/serve.retrieval.yaml, vllm/serve.gemma.yaml)
LLMS: dict[str, dict] = {
    "qwen": {"label": "qwen3.8-27b-int4", "model_id": "RedHatAI/Qwen3.8-27B-INT4", "concurrency": 4},
    "gemma": {"label": "gemma-4-26b-a4b-qat-int4", "model_id": "cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4", "concurrency": 4},
}

# Thinking is always off (chat_template_kwargs enable_thinking false: Qwen's switch, and Gemma 4's default).
DEFAULT_PARAMS = {"temperature": 0.0, "max_tokens": 700, "seed": 0}

# The commands that start each model, for the error messages (the user runs them in a terminal).
START_COMMANDS = {
    "qwen": "docker compose stop vllm encoder-gpu && docker compose up -d vllm",
    "gemma": ("docker compose stop vllm && VLLM_MODEL_ID=cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4 "
              "VLLM_CONFIG=serve.gemma.yaml docker compose up -d vllm"),
}


def base_url() -> str:
    return os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1")


def reload_prompts():
    """Re-read ``rag_prompts.py`` (so an edit is used without restarting the kernel)."""
    return importlib.reload(rag_prompts)


def combo_path(out_dir: str | Path, llm_key: str, prompt: str) -> Path:
    """``<out_dir>/rag_<llm label>__<prompt>.jsonl``"""
    return Path(out_dir) / f"rag_{LLMS[llm_key]['label']}__{prompt}.jsonl"


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


# ---------------------------------------------------------------- the server


def served_models(url: str | None = None) -> list[str]:
    """The model ids vLLM serves now (``GET /v1/models``); an empty list when it does not answer."""
    try:
        with urllib.request.urlopen((url or base_url()).rstrip("/") + "/models", timeout=10) as r:
            return [m["id"] for m in json.load(r)["data"]]
    except Exception:
        return []


def wait_for_model(llm_key: str, timeout_s: float = 60, strict: bool = True, poll_s: float = 5) -> str:
    """Wait until vLLM serves the model of ``llm_key``; returns its id.

    ``strict``: raise at once when vLLM serves another model (a run cell that would write records
    with the wrong model). Without it, keep waiting through that (the switch cell: the old container
    may still be going down). Raises with the commands to start the model when the time is up."""
    want = LLMS[llm_key]["model_id"]
    t0 = time.time()
    while True:
        have = served_models()
        if want in have:
            return want
        if have and strict:
            raise RuntimeError(f"vLLM serves {have}, but this cell needs {want!r} ({llm_key}). Start it in a terminal:\n  {START_COMMANDS[llm_key]}")
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"vLLM does not serve {want!r} after {timeout_s:.0f} s (serves: {have or 'nothing, not answering'}). "
                               f"Start it in a terminal:\n  {START_COMMANDS[llm_key]}\n  docker compose logs --tail 50 vllm")
        time.sleep(poll_s)


@lru_cache(maxsize=4)
def _client(url: str, key: str):
    from openai import OpenAI

    return OpenAI(base_url=url, api_key=key, timeout=900.0, max_retries=0)


def generate(model_id: str, messages: list[dict], params: dict | None = None, retries: int = 3) -> dict:
    """One chat completion: ``content``, ``finish_reason``, ``prompt_tokens``, ``completion_tokens``, ``seconds``.

    Retries a few times, with a growing pause, on connection errors, timeouts and server errors; any other
    error (for example a prompt longer than the model's context) is raised at once."""
    import openai

    p = {**DEFAULT_PARAMS, **(params or {})}
    client = _client(base_url(), os.environ.get("VLLM_API_KEY", "EMPTY"))
    transient = (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError, openai.RateLimitError)
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=model_id, messages=messages, temperature=p["temperature"], max_tokens=p["max_tokens"], seed=p["seed"],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        except transient:
            if attempt == retries:
                raise
            time.sleep(5 * (attempt + 1))
            continue
        c = r.choices[0]
        return {"content": (c.message.content or "").strip(), "finish_reason": c.finish_reason,
                "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens,
                "seconds": round(time.time() - t0, 3)}
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- inputs


def load_inputs(exp, contexts_path: str | Path) -> list[dict]:
    """The questions of the experiment, each joined with its X3 context: ``[{"question": <question record>,
    "context": <contexts.jsonl line>}, ...]`` in question order. Raises when the two do not cover the same ids."""
    questions = exp.read_questions()
    ctx = {r["question_id"]: r for r in read_jsonl(contexts_path)}
    ids = [q["question_id"] for q in questions]
    missing, extra = [i for i in ids if i not in ctx], sorted(set(ctx) - set(ids))
    if missing or extra:
        raise ValueError(f"{contexts_path} does not match the questions of experiment {exp.name!r}: "
                         f"{len(missing)} without a context ({missing[:3]}), {len(extra)} contexts without a question ({extra[:3]}); "
                         "run notebook 10 for this experiment")
    for q in questions:
        if ctx[q["question_id"]]["question"] != q["question"]:
            raise ValueError(f"{q['question_id']}: the question text differs between the question file and {contexts_path}")
    return [{"question": q, "context": ctx[q["question_id"]]} for q in questions]


def make_record(item: dict, llm_key: str, prompt: str, params: dict, result: dict | None, error: str | None = None) -> dict:
    """One output record (x3_to_llm.md 4.1). Gold and reference answer come from the question file and are never in the prompt."""
    q, c = item["question"], item["context"]
    return {
        "question_id": q["question_id"],
        "question": q["question"],
        "family": c.get("family"),
        "answerable": bool(q["answerable"]),
        "llm": LLMS[llm_key]["label"],
        "model_id": LLMS[llm_key]["model_id"],
        "prompt": prompt,
        "prompt_hash": rag_prompts.prompt_hash(prompt),
        "answer": result["content"] if result else "",
        "gold": [{"section_id": g["section_id"], "role": g["role"], "quote": g["quote"]} for g in q["gold"]],
        "reference_answer": q.get("answer"),
        "answer_aliases": q.get("answer_aliases", []),
        "distractor_section_ids": q.get("distractor_section_ids", []),
        "retrieved": c["hits"],
        "retrieval_run_id": c.get("run_id"),
        "engine_type_detected": c.get("engine_type_detected"),
        "context_tokens": c.get("context_tokens"),
        "finish_reason": result["finish_reason"] if result else "error",
        "prompt_tokens": result["prompt_tokens"] if result else None,
        "completion_tokens": result["completion_tokens"] if result else None,
        "seconds": result["seconds"] if result else None,
        "params": {**params, "thinking": False},
        "error": error,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------- one question, to see what happens


def try_question(items: list[dict], index: int = 0, params: dict | None = None) -> dict:
    """Ask the model that vLLM serves now the question at ``index`` with every prompt. Writes nothing.

    Returns ``{"llm", "model_id", "question_id", "question", "gold", "reference_answer", "prompts": {name: {"text":
    the prompt as the model sees it, "answer", "prompt_tokens", "completion_tokens", "seconds", "finish_reason"}}}``."""
    reload_prompts()
    have = served_models()
    key = next((k for k, v in LLMS.items() if v["model_id"] in have), None)
    if key is None:
        raise RuntimeError(f"vLLM serves {have or 'nothing (not answering)'}, which is none of {[v['model_id'] for v in LLMS.values()]}. "
                           f"Start one:\n  {START_COMMANDS['qwen']}\n  {START_COMMANDS['gemma']}")
    item = items[index]
    q, hits = item["question"], item["context"]["hits"]
    out = {"llm": LLMS[key]["label"], "model_id": LLMS[key]["model_id"], "question_id": q["question_id"], "question": q["question"],
           "gold": [g["section_id"] for g in q["gold"]], "reference_answer": q.get("answer"), "prompts": {}}
    for name in rag_prompts.PROMPTS:
        r = generate(LLMS[key]["model_id"], rag_prompts.build_messages(name, q["question"], hits), params)
        out["prompts"][name] = {"text": rag_prompts.show(name, q["question"], hits), "answer": r["content"], "prompt_tokens": r["prompt_tokens"],
                                "completion_tokens": r["completion_tokens"], "seconds": r["seconds"], "finish_reason": r["finish_reason"]}
    return out


# ---------------------------------------------------------------- one (LLM, prompt) over all questions


def run_combination(items: list[dict], llm_key: str, prompt: str, out_path: str | Path, params: dict | None = None,
                    limit: int | None = None, progress=None, wait_s: float = 60) -> dict:
    """Answer every question with one LLM and one prompt; append each record to ``out_path`` as it is done.

    - It first checks that vLLM serves this LLM (``wait_for_model``, strict).
    - **Resumable:** records already in the file (without an error) are kept and their questions skipped; records with
      an error are redone. A file that holds records of another prompt version (``prompt_hash``), prompt or model is an
      error: the prompt was edited, or the file belongs to something else; delete the file or rename the prompt.
    - A question whose request fails is written with ``error`` set and the run goes on.
    - ``limit`` runs only the first questions (a smoke test; delete the file before the full run).
    ``progress`` is a tqdm-like callable. Returns the counts."""
    reload_prompts()
    llm = LLMS[llm_key]
    p = {**DEFAULT_PARAMS, **(params or {})}
    ph = rag_prompts.prompt_hash(prompt)
    wait_for_model(llm_key, timeout_s=wait_s, strict=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(out_path) if out_path.is_file() else []
    for r in existing:
        if (r["prompt"], r["prompt_hash"], r["model_id"]) != (prompt, ph, llm["model_id"]):
            raise ValueError(f"{out_path} holds a record of prompt {r['prompt']!r} (hash {r['prompt_hash']}) and model {r['model_id']!r}; "
                             f"this run is prompt {prompt!r} (hash {ph}) and model {llm['model_id']!r}. "
                             "The prompt was edited or the file belongs to another run: delete the file or rename the prompt.")
    good = [r for r in existing if not r.get("error")]
    if len(good) != len(existing):  # drop the failed ones; they are redone below
        with open(out_path, "w", encoding="utf-8") as f:
            for r in good:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    done = {r["question_id"] for r in good}
    todo = [it for it in (items[:limit] if limit else items) if it["question"]["question_id"] not in done]

    def one(item):
        q = item["question"]
        try:
            res = generate(llm["model_id"], rag_prompts.build_messages(prompt, q["question"], item["context"]["hits"]), p)
            return make_record(item, llm_key, prompt, p, res)
        except Exception as e:  # recorded, not raised: one bad request must not stop 477 questions
            return make_record(item, llm_key, prompt, p, None, error=f"{type(e).__name__}: {e}")

    t0 = time.time()
    n_new = n_err = 0
    with ThreadPoolExecutor(max_workers=llm["concurrency"]) as pool, open(out_path, "a", encoding="utf-8") as f:
        futures = [pool.submit(one, it) for it in todo]
        for fut in (progress(as_completed(futures), total=len(futures), desc=f"{llm['label']} x {prompt}") if progress else as_completed(futures)):
            rec = fut.result()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            n_err += bool(rec["error"])
    records = read_jsonl(out_path)
    return {"file": str(out_path), "records": len(records), "new": n_new, "skipped": len(good), "errors": n_err,
            "truncated": sum(r["finish_reason"] == "length" for r in records), "seconds": round(time.time() - t0)}


# ---------------------------------------------------------------- checks, combine, summary


def check_combination(path: str | Path, items: list[dict], llm_key: str, prompt: str) -> list[str]:
    """Checks 1-4 of x3_to_llm.md section 8 for one file; returns the problems (an empty list is a pass).

    1. one record per question of ``items``, no duplicates, no error; 2. one model, prompt and prompt hash, and the
    prompt as it is in ``rag_prompts.py`` now; 3. ``retrieved`` equals the contexts' hits and ``gold`` the question
    file's; 4. no empty answer and no truncated one (``finish_reason`` length)."""
    bad: list[str] = []
    if not Path(path).is_file():
        return [f"{path} does not exist"]
    recs = read_jsonl(path)
    ids = [it["question"]["question_id"] for it in items]
    got = [r["question_id"] for r in recs]
    dup = sorted({i for i in got if got.count(i) > 1})
    if dup:
        bad.append(f"duplicate questions: {dup[:3]}")
    if set(ids) - set(got):
        bad.append(f"{len(set(ids) - set(got))} questions without a record, e.g. {sorted(set(ids) - set(got))[:3]}")
    if set(got) - set(ids):
        bad.append(f"{len(set(got) - set(ids))} records of unknown questions, e.g. {sorted(set(got) - set(ids))[:3]}")
    errors = [r["question_id"] for r in recs if r.get("error")]
    if errors:
        bad.append(f"{len(errors)} records with an error, e.g. {errors[:3]}: run the cell again to redo them")
    want = (LLMS[llm_key]["label"], LLMS[llm_key]["model_id"], prompt, rag_prompts.prompt_hash(prompt))
    other = [r["question_id"] for r in recs if (r["llm"], r["model_id"], r["prompt"], r["prompt_hash"]) != want]
    if other:
        bad.append(f"{len(other)} records of another model, prompt or prompt version, e.g. {other[:3]} (expected {want})")
    by_id = {it["question"]["question_id"]: it for it in items}
    drift = [r["question_id"] for r in recs if r["question_id"] in by_id and (
        r["retrieved"] != by_id[r["question_id"]]["context"]["hits"]
        or r["gold"] != [{"section_id": g["section_id"], "role": g["role"], "quote": g["quote"]} for g in by_id[r["question_id"]]["question"]["gold"]])]
    if drift:
        bad.append(f"{len(drift)} records whose retrieved parts or gold differ from the inputs, e.g. {drift[:3]}")
    empty = [r["question_id"] for r in recs if not r.get("error") and not r["answer"].strip()]
    if empty:
        bad.append(f"{len(empty)} empty answers, e.g. {empty[:3]}")
    cut = [r["question_id"] for r in recs if r["finish_reason"] == "length"]
    if cut:
        bad.append(f"{len(cut)} answers cut off by max_tokens, e.g. {cut[:3]}: raise max_tokens and delete the file")
    return bad


def combine(out_dir: str | Path, items: list[dict], llm_order: list[str], prompt_order: list[str], out_path: str | Path) -> int:
    """Write ``out_path``: the combination files in (LLM, prompt) order, each in question order. Returns the number of records."""
    order = {it["question"]["question_id"]: i for i, it in enumerate(items)}
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for llm_key in llm_order:
            for prompt in prompt_order:
                for r in sorted(read_jsonl(combo_path(out_dir, llm_key, prompt)), key=lambda r: order[r["question_id"]]):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
    return n


def check_combined(out_path: str | Path, out_dir: str | Path, items: list[dict], llm_order: list[str], prompt_order: list[str]) -> list[str]:
    """Check 5: 4 x (number of questions) lines, every (question, LLM, prompt) once, equal to the four files put together."""
    recs = read_jsonl(out_path)
    expected = []
    for llm_key in llm_order:
        for prompt in prompt_order:
            order = {it["question"]["question_id"]: i for i, it in enumerate(items)}
            expected += sorted(read_jsonl(combo_path(out_dir, llm_key, prompt)), key=lambda r: order[r["question_id"]])
    bad = []
    if len(recs) != len(llm_order) * len(prompt_order) * len(items):
        bad.append(f"{len(recs)} records, expected {len(llm_order)} x {len(prompt_order)} x {len(items)}")
    keys = [(r["question_id"], r["llm"], r["prompt"]) for r in recs]
    if len(set(keys)) != len(keys):
        bad.append("a (question, LLM, prompt) occurs more than once")
    if recs != expected:
        bad.append("the combined file is not the four files put together, in order")
    return bad


def summary(out_dir: str | Path, llm_order: list[str], prompt_order: list[str]):
    """One row per (LLM, prompt): records, answerable and unanswerable, answers that start with the abstention sentence
    (``rag_prompts.NO_ANSWER``) among each, truncated answers, errors, mean tokens and seconds."""
    import pandas as pd

    rows = []
    for llm_key in llm_order:
        for prompt in prompt_order:
            path = combo_path(out_dir, llm_key, prompt)
            if not path.is_file():
                continue
            recs = read_jsonl(path)
            ans = [r for r in recs if r["answerable"]]
            un = [r for r in recs if not r["answerable"]]
            starts = lambda rs: sum(r["answer"].strip().startswith(rag_prompts.NO_ANSWER) for r in rs)  # noqa: E731
            ok = [r for r in recs if not r.get("error")]
            mean = lambda k: round(sum(r[k] for r in ok) / len(ok), 2) if ok else float("nan")  # noqa: E731
            rows.append({"llm": LLMS[llm_key]["label"], "prompt": prompt, "records": len(recs), "answerable": len(ans), "unanswerable": len(un),
                         "abstained_answerable": starts(ans), "abstained_unanswerable": starts(un),
                         "truncated": sum(r["finish_reason"] == "length" for r in recs), "errors": sum(bool(r.get("error")) for r in recs),
                         "prompt_tokens": mean("prompt_tokens"), "completion_tokens": mean("completion_tokens"), "seconds": mean("seconds")})
    return pd.DataFrame(rows)
