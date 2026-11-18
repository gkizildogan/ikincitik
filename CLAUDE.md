# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research project. It has two parts:
1. **Extraction:** notebooks 00–03 and `src/csparse/` turn `cs-e_amendment_9.pdf` (EASA CS-E Amendment 9) into section-bounded Markdown retrieval units, using Qwen as a VLM.
2. **Retrieval experiments:** notebooks 04–10 and `src/csrag/` compare retrieval methods over that corpus, and notebook 10 runs the final pipeline X3 (`x2_to_x3_plan.md`, gitignored like `plan.md`). Notebook 11 generates answers from the X3 contexts with two LLMs × two prompts (`x3_to_llm.md`, gitignored) and writes the final output `results/<experiment>/llm/rag_output.jsonl`. Question set 1 has 100 questions (92 answerable, 8 unanswerable); question set 2 (`data/eval/set2/`) adds 377 answerable questions in four families. Which files an evaluation runs, and where its results go, is set by the experiment in `experiments.toml`.

`README.md` is the reference for services, files, record formats and results. `plan.md` is the design document; it is gitignored but present locally. Section numbers in code comments ("plan 4.8", "plan section 5") refer to `plan.md`. When behaviour changes, update both documents, because the user keeps them in sync with the code.

`ignore_this_pdf.pdf` is unrelated; ignore it.

To add more documents (e.g. CS-27) or update stored ones for production, follow `adding_new_docs.md`. Never write to the experiment collections `cse_units_v1` / `cse_win500_v1`.

## Commands

```bash
uv sync                                    # deps; csparse and csrag are installed from src/ (uv_build, two modules)
uv run jupyter lab                         # kernel "ikincitik (uv)"; kernel name: ikincitik
# headless notebook run (from notebooks/); CSRAG_EXPERIMENT=<name> in front overrides `active` in experiments.toml:
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=ikincitik --ExecutePreprocessor.timeout=-1 07_run_retrievers.ipynb
# lint (no config in pyproject; this is the invocation used so far):
uvx ruff@0.14.0 check --select F,E9 --line-length 140 src encoder
```

There is no test suite. Verification happens in the notebooks: each one ends with assertion-style checks (coverage, ingestion round-trip, ANN vs exact, RRF vs Qdrant RRF, ColBERT engine agreement, reproducibility, label leakage). Run the relevant notebook to verify a change.

Notebooks `chdir` to the repo root in their first cell, so every path in `src/` is relative to the repo root (`data/`, `results/`, `experiments.toml`). The notebooks only call into `src/` and expose parameters; put logic in `src/`, not in notebook cells.

## GPU and services (single RTX 3090, ~21 GB usable)

`compose.yaml` runs `vllm` (Qwen3.8-27B-INT4), `qdrant`, `encoder` (CPU) and `encoder-gpu` (profile `gpu`). Both encoders publish port 8081, so only one can run at a time. **vLLM and `encoder-gpu` never run together.** The experiments run in two phases:

```bash
# phase A, notebooks 04-07 and 10: GPU encoder, vLLM stopped
docker compose stop vllm encoder && docker compose up -d qdrant && docker compose up -d --force-recreate encoder-gpu
# phase B, notebook 08: vLLM + CPU encoder
docker compose stop encoder-gpu && docker compose up -d qdrant vllm && docker compose up -d --force-recreate encoder
# notebook 11 (generation) reads results/<experiment>/x3/contexts.jsonl and needs only vLLM, one model at a time:
docker compose stop vllm encoder-gpu && docker compose up -d vllm      # Qwen
docker compose stop vllm && VLLM_MODEL_ID=cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4 VLLM_CONFIG=serve.gemma.yaml docker compose up -d vllm   # Gemma
```

- Encoders need `--force-recreate`: a failed start (port taken) leaves a container with no published port.
- Other vLLM stacks on this machine (skyrag-vllm, aviation-vllm*) must be stopped first.
- `VLLM_CONFIG` selects the serve file: `vllm/serve.yaml` is the extraction config (vision) and is kept unchanged for reproducibility; `vllm/serve.retrieval.yaml` is the experiments config (text-only, 32k context, `gpu-memory-utilization 0.80`). Don't raise the memory setting: 0.85 OOMs next to the desktop's ~3 GB.
- `compose.yaml` takes the vLLM model from `VLLM_MODEL_ID` (default Qwen3.8-27B-INT4); `vllm/serve.gemma.yaml` serves Gemma 4 26B-A4B (QAT 4-bit, `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4`) under the same settings as Qwen. Gemma 4 31B does not fit next to the desktop (vLLM allocates a throw-away 2.6 GiB `lm_head` while building it); don't pull it again.
- Keep n-gram speculative decoding off; it corrupts greedy output on this model.
- `encoder/` is a separate Docker image (FastAPI: `/embed`, `/rerank`, `/tokens`, `/health`) with its own fully pinned `requirements.txt`, compiled from `requirements.in`. Models load offline from pinned snapshot paths. Its dependencies are not in `pyproject.toml`.

## Experiments and results folders

- An experiment is a name plus question files, declared in `experiments.toml` (`[experiments.<name>]`: `questions`, optional `description` and `qrels`). `src/csrag/experiment.py` loads it. **Everything it produces goes to `results/<name>/`**: `runs/`, `qrels.jsonl`, the metrics files and `diagrams/`. Never write results directly into `results/`.
- The active experiment is `active` in `experiments.toml`; the `CSRAG_EXPERIMENT` environment variable (shell or `.env`) overrides it. Notebooks 06–11 get it from `experiment.active()` in their first cell (`EXP`, `RESULTS = EXP.dir`) and don't name a folder themselves. `runs.py` and `report.py` default to the active experiment's folder, so pass `root=` / `out_root=` only to work somewhere else (the reproducibility check in notebook 08 does).
- The experiment name is not `PARAMS_VERSION`. One experiment holds the v1 runs and, later, the v2 runs and the X3 run (v3) of the same questions. A new question set gets a new experiment name; don't add question files to an experiment that already has runs.
- Run ids don't depend on the questions. `config.json` therefore records `n_questions` and `questions_hash`, and `runs.execute` raises when a completed run was made with other questions (`runs.same_questions`). Don't weaken this check.
- Defined now: `run1` (question set 1; qrels stay in `data/eval/qrels.jsonl`) and `run2` (set 1 + the four set 2 files, 477 questions, 469 answerable). The `run1` results were moved out of `results/` into `run1resultarchive/` (gitignored); `mv run1resultarchive results/run1` restores them, and notebook 09 then reproduces the archived tables byte for byte. Use this as the regression check after changing `evaluate.py` or `report.py`, on a copy, without overwriting the archive.
- Question families: `family` in a question record, `set1` when absent (`evaluate.family_of`). With more than one family, `evaluate_all` adds `family=<name>` subsets (with CIs) and `report.py` adds per-family columns; the `by_family` sheet is always written. The statistics still treat questions as independent, although a paraphrase shares its gold with its original, so pooled CIs and tests are too narrow.
- Nothing in the code may hardcode the number of questions (92, 8, 100): take it from the questions.
- **`llm/` is notebook 11's folder** (`rag_<llm>__<prompt>.jsonl`, `rag_output.jsonl`, `summary.csv`); notebook 09 does not write it. **`rag_output.jsonl` is the final output of the whole pipeline**: one record per (question, LLM, prompt), the four combination files put together in `LLM_ORDER` × `PROMPT_ORDER` × question order (`run2`: 477 × 4 = 1,908 lines). Its record format is in README ("Generation experiments"). Only the combine cell writes it, from the four combination files; they are the record (there is no cache for generation), so never delete or hand-edit them except as described under *Generation* below.
- **`x3/` is notebook 10's folder.** The X3 run record is in `results/<experiment>/runs/` like every run, and everything else the final pipeline produces (`contexts.jsonl`, its own copies of the standard metric files for X3 and the baselines, the X2 vs X3 tables, the cut-off sweep, the keyword report, its diagram) is in `results/<experiment>/x3/`. Notebook 09 does not write it. `final.write_metric_files` uses the same code as notebook 09, so X3's rows equal what notebook 09 would give.

## csrag architecture

- **Config-driven pipeline.** `retrievers/base.py` defines every method (L1, S1, D1, C1, H1, H2, M1, F0, F1, T1, Q1, Q2, X1, X2, T0, T2, T3, T2+T3, T4) as a plain config dict from `method_config()`. `grid()` expands these across index × gate × budget. A single `Pipeline` in `retrievers/__init__.py` interprets any config in this order: query_transform → first stages → gate A → fusion → filter → ColBERT rerank / rerank → gate B → lookup → expansion. To add a method, add a config branch. Don't add a new retriever class. **The config of an existing method must not change by a single key**: `config_hash` covers the whole dict, so a new key in the shared base dict of `method_config` would rename all 110 stored v1 runs. New keys go into the new method's branch only (notebook 10 checks that the v1 grid ids are unchanged).
- **X3, the final pipeline.** X3 (H1 → keyword engine-type filter → jina on 30 candidates → score cut-off, windows, gate off) is built with `method_config("X3", ...)` and params v3, and is in `FINAL_METHODS`, not `ALL_METHODS`: `grid()` refuses v3 (it has a `cut` entry) and notebooks 07–08 never see X3. The keyword rule is `csrag/engine_rule.py` reading `engine_keywords.toml`, which the user edits by hand; its hash is part of the config, so an edited file is a new run id. `final.py` holds the logic of notebook 10 so that `evaluate.py` and `report.py` keep reproducing the run 1 archive.
- **Run identity.** The config, including the hashes of the prompts from `query_llm.prompt_hashes()`, is hashed into `config_hash`. The run id is `<method>[-b<budget>]-<index>-gate<on|off>-<params_version>-<hash8>`. Editing a prompt or a parameter therefore creates new runs instead of overwriting old ones.
- **Parameters are frozen.** `PARAMS["v1"]` must not be tuned. The plan allows exactly one hand revision, `PARAMS["v2"] = revise("v1", ...)`, after reading all v1 metrics. That revision is now written (`rrf_k`, gate A, gate B, the agent's starting hits; reasons in `plan.md` 4.8), so `PARAMS["v2"]` must not be tuned either. `PARAMS["v3"]` is X3's parameter set (built on v1; `rrf_k` 10, 30 candidates, `k` 8, a `cut` entry; `plan.md` 4.9) and is frozen once X3 has run. The user decides when a new version is made; never edit a version that has runs. All versions are reported.
- **Labels never reach retrievers.** `retrieve(question_id, text, k)` receives only the question text. The one exception is F0, the oracle filter, which gets `engine_type` labels at construction. X3's keyword rule reads the question text and `engine_keywords.toml` only. Notebooks 08 and 10 check for label leakage.
- **Caches make reruns exact and resumable.** `runs.py` defines append-only JSONL caches in `data/cache/` (`llm`, `search`, `rerank_bge`, `rerank_jina`; last write wins), and they are committed to git. They are shared by all experiments. `runs.execute` skips completed runs and resumes from `results/<experiment>/runs/<id>/partial.jsonl`. LLM calls are keyed by a hash of the full request.
- **Two indexes over the same text:** `units` → Qdrant `cse_units_v1` (252 base units from `chunks.jsonl`) and `win500` → `cse_win500_v1` (436 windows of 500 bge-m3 tokens). Evaluation is span-based (`evaluate.py`), so both indexes are scored against the same gold character spans, at unit, section and group level.
- **ColBERT runs client-side.** Qdrant caps a multivector at 1023 vectors, and 60 base units exceed that. C1 and M1 therefore compute MaxSim client-side (`ColbertIndex`) from the cached vectors in `data/embeddings/`, for both indexes.
- **Generation (notebook 11).** 2 LLMs (`qwen`, `gemma` in `rag_generate.LLMS`) × 2 prompts (`simple`, `robust` in `rag_prompts.PROMPTS`) over every question of the active experiment, from `results/<experiment>/x3/contexts.jsonl`. `rag_prompts.py` holds the prompts (the user edits the texts there; `prompt_hash` is stored in every record), `rag_generate.py` the vLLM client and the resumable per-(LLM, prompt) runs. The notebook needs only vLLM, never Qdrant or the encoder, and each run cell checks that vLLM serves *its* model (strict, via `/v1/models`) before writing anything. Rules to keep:
  - **The prompt sees only the question and the retrieved windows.** `gold`, `reference_answer`, `answer_aliases` and `distractor_section_ids` are copied into the record from the question file and never into a prompt. All questions are asked, answerable or not.
  - **Same request for every combination:** temperature 0, seed 0, thinking off (`chat_template_kwargs: {enable_thinking: False}`, Gemma and Qwen alike), `max_tokens` from `PARAMS` in the notebook. No cache: generation is not exact to the byte, so the JSONL files are the record.
  - **Resume:** `run_combination` skips questions that have a record and redoes only records with `error`. It raises if a file holds another `prompt`, `prompt_hash` or `model_id`, so an edited prompt needs a new file (delete it) or a new prompt name. It does not compare `params`: each record stores its own `params`, and records of one file may differ in `max_tokens` after a redo.
  - **Cut-off answers** (`finish_reason == "length"`) are not redone by the run cell and fail check 4 of the combine cell. The fix is manual: delete those lines from the combination file, raise `max_tokens`, run the parameter cell (cells read `PARAMS` from the kernel), run the combination cell with its model served. `run2` went through this: its records carry `max_tokens` 1000, 2000 or 4000.
  - **`summary.csv`:** `abstained_*` counts answers that *start with* `rag_prompts.NO_ANSWER`, a text match that is 0 by construction for `simple` (it never asks for the sentence). It is not a quality measure.
  - Generation results are not written into the documents (result write-ups are about retrieval); the documents describe notebook 11's design, files and record format only.
- **Corpus in memory.** `csrag.Corpus` extends `csparse.merge.Corpus`. Expansion (T2, T3, T4) fetches parent sections and cross-referenced sections from memory, not from Qdrant.
- **Notebook split by phase.** Notebook 07 (phase A) runs everything that needs no LLM, including the X1/X2 rerankers, and asserts that vLLM is down. Notebook 08 (phase B) runs Q1, Q2, F1 and T4 and needs vLLM. Notebook 09 needs no models and writes every file under `results/<experiment>/` except `runs/`, `qrels.jsonl` (notebook 06), `x3/` and `llm/`. Notebook 10 (phase A again: `qdrant` + `encoder-gpu`, vLLM stopped, no LLM) runs X3 and writes `x3/`; with `SERVICES = False` it works from the caches alone. Notebook 11 (vLLM only, one model at a time) runs after notebook 10, reads `x3/contexts.jsonl` and writes `llm/`. Notebooks 04 and 05 are the corpus side and don't depend on the experiment.

## csparse notes

- VLM output is accepted only if it matches the PDF text layer word for word (recall and precision ≥ 0.995, no missing numbers). Otherwise it retries once, then falls back to the text layer. `rescore_cache()` re-applies changed scoring rules without calling the model.
- List nesting is rebuilt from the markers in `merge.indent_markers`, which handles CS-E's 5-level `(a)(1)(i)(A)(1)` scheme and the ambiguity between `(i)`, `(v)` and `(x)` as letters or roman numerals. It is not taken from the model.
- `data/sections.jsonl`, `groups.jsonl`, `chunks.jsonl`, `windows_500.jsonl`, `cross_refs.jsonl` and `eval/qrels.jsonl` (the qrels of question set 1; other experiments keep theirs in `results/<experiment>/qrels.jsonl`) are deterministic outputs committed to git. `data/vlm_cache/` and `data/embeddings/` are gitignored and can be regenerated.
