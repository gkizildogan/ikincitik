# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research project. It has two parts:
1. **Extraction:** notebooks 00–03 and `src/csparse/` turn `cs-e_amendment_9.pdf` (EASA CS-E Amendment 9) into section-bounded Markdown retrieval units, using Qwen as a VLM.
2. **Retrieval experiments:** notebooks 04–09 and `src/csrag/` compare retrieval methods over that corpus on 100 questions (92 answerable, 8 unanswerable).

`README.md` is the reference for services, files, record formats and results. `plan.md` is the design document; it is gitignored but present locally. Section numbers in code comments ("plan 4.8", "plan section 5") refer to `plan.md`. When behaviour changes, update both documents, because the user keeps them in sync with the code.

`ignore_this_pdf.pdf` is unrelated; ignore it.

To add more documents (e.g. CS-27) or update stored ones for production, follow `adding_new_docs.md`. Never write to the experiment collections `cse_units_v1` / `cse_win500_v1`.

## Commands

```bash
uv sync                                    # deps; csparse and csrag are installed from src/ (uv_build, two modules)
uv run jupyter lab                         # kernel "ikincitik (uv)"; kernel name: ikincitik
# headless notebook run (from notebooks/):
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=ikincitik --ExecutePreprocessor.timeout=-1 07_run_retrievers.ipynb
# lint (no config in pyproject; this is the invocation used so far):
uvx ruff@0.14.0 check --select F,E9 --line-length 140 src encoder
```

There is no test suite. Verification happens in the notebooks: each one ends with assertion-style checks (coverage, ingestion round-trip, ANN vs exact, RRF vs Qdrant RRF, ColBERT engine agreement, reproducibility, label leakage). Run the relevant notebook to verify a change.

Notebooks `chdir` to the repo root in their first cell, so every path in `src/` is relative to the repo root (`data/`, `results/`). The notebooks only call into `src/` and expose parameters; put logic in `src/`, not in notebook cells.

## GPU and services (single RTX 3090, ~21 GB usable)

`compose.yaml` runs `vllm` (Qwen3.8-27B-INT4), `qdrant`, `encoder` (CPU) and `encoder-gpu` (profile `gpu`). Both encoders publish port 8081, so only one can run at a time. **vLLM and `encoder-gpu` never run together.** The experiments run in two phases:

```bash
# phase A, notebooks 04-07: GPU encoder, vLLM stopped
docker compose stop vllm encoder && docker compose up -d qdrant && docker compose up -d --force-recreate encoder-gpu
# phase B, notebook 08: vLLM + CPU encoder
docker compose stop encoder-gpu && docker compose up -d qdrant vllm && docker compose up -d --force-recreate encoder
```

- Encoders need `--force-recreate`: a failed start (port taken) leaves a container with no published port.
- Other vLLM stacks on this machine (skyrag-vllm, aviation-vllm*) must be stopped first.
- `VLLM_CONFIG` selects the serve file: `vllm/serve.yaml` is the extraction config (vision) and is kept unchanged for reproducibility; `vllm/serve.retrieval.yaml` is the experiments config (text-only, 32k context, `gpu-memory-utilization 0.80`). Don't raise the memory setting: 0.85 OOMs next to the desktop's ~3 GB.
- Keep n-gram speculative decoding off; it corrupts greedy output on this model.
- `encoder/` is a separate Docker image (FastAPI: `/embed`, `/rerank`, `/tokens`, `/health`) with its own fully pinned `requirements.txt`, compiled from `requirements.in`. Models load offline from pinned snapshot paths. Its dependencies are not in `pyproject.toml`.

## csrag architecture

- **Config-driven pipeline.** `retrievers/base.py` defines every method (L1, S1, D1, C1, H1, H2, M1, F0, F1, T1, Q1, Q2, X1, X2, T0, T2, T3, T2+T3, T4) as a plain config dict from `method_config()`. `grid()` expands these across index × gate × budget. A single `Pipeline` in `retrievers/__init__.py` interprets any config in this order: query_transform → first stages → gate A → fusion → filter → ColBERT rerank / rerank → gate B → lookup → expansion. To add a method, add a config branch. Don't add a new retriever class.
- **Run identity.** The config, including the hashes of the prompts from `query_llm.prompt_hashes()`, is hashed into `config_hash`. The run id is `<method>[-b<budget>]-<index>-gate<on|off>-<params_version>-<hash8>`. Editing a prompt or a parameter therefore creates new runs instead of overwriting old ones.
- **Parameters are frozen.** `PARAMS["v1"]` must not be tuned. The plan allows exactly one hand revision, `PARAMS["v2"] = revise("v1", ...)`, after reading all v1 metrics. Both versions are reported.
- **Labels never reach retrievers.** `retrieve(question_id, text, k)` receives only the question text. The one exception is F0, the oracle filter, which gets `engine_type` labels at construction. Notebook 08 checks for label leakage.
- **Caches make reruns exact and resumable.** `runs.py` defines append-only JSONL caches in `data/cache/` (`llm`, `search`, `rerank_bge`, `rerank_jina`; last write wins), and they are committed to git. `runs.execute` skips completed runs and resumes from `results/runs/<id>/partial.jsonl`. LLM calls are keyed by a hash of the full request.
- **Two indexes over the same text:** `units` → Qdrant `cse_units_v1` (252 base units from `chunks.jsonl`) and `win500` → `cse_win500_v1` (436 windows of 500 bge-m3 tokens). Evaluation is span-based (`evaluate.py`), so both indexes are scored against the same gold character spans, at unit, section and group level.
- **ColBERT runs client-side.** Qdrant caps a multivector at 1023 vectors, and 60 base units exceed that. C1 and M1 therefore compute MaxSim client-side (`ColbertIndex`) from the cached vectors in `data/embeddings/`, for both indexes.
- **Corpus in memory.** `csrag.Corpus` extends `csparse.merge.Corpus`. Expansion (T2, T3, T4) fetches parent sections and cross-referenced sections from memory, not from Qdrant.
- **Notebook split by phase.** Notebook 07 (phase A) runs everything that needs no LLM, including the X1/X2 rerankers, and asserts that vLLM is down. Notebook 08 (phase B) runs Q1, Q2, F1 and T4 and needs vLLM. Notebook 09 needs no models and writes every file under `results/` except `runs/`.

## csparse notes

- VLM output is accepted only if it matches the PDF text layer word for word (recall and precision ≥ 0.995, no missing numbers). Otherwise it retries once, then falls back to the text layer. `rescore_cache()` re-applies changed scoring rules without calling the model.
- List nesting is rebuilt from the markers in `merge.indent_markers`, which handles CS-E's 5-level `(a)(1)(i)(A)(1)` scheme and the ambiguity between `(i)`, `(v)` and `(x)` as letters or roman numerals. It is not taken from the model.
- `data/sections.jsonl`, `groups.jsonl`, `chunks.jsonl`, `windows_500.jsonl`, `cross_refs.jsonl` and `eval/qrels.jsonl` are deterministic outputs committed to git. `data/vlm_cache/` and `data/embeddings/` are gitignored and can be regenerated.
