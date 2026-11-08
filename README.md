# CS-E Amendment 9 → section-bounded Markdown for RAG, and retrieval experiments

Two parts:

1. **Extraction (notebooks 00–03, `src/csparse/`).** Turns `cs-e_amendment_9.pdf` (EASA CS-E Amendment 9, with the header and footer cropped off) into retrieval units. There is one unit per `CS-E X` and one per `AMC E X…`, with lists, glossaries and tables kept intact across page breaks. `[Amdt No: …]` notes are removed and figures are skipped.
2. **Retrieval experiments (notebooks 04–09, `src/csrag/`).** Compares retrieval methods for single-turn question answering over that text: lexical, learned sparse, dense, late interaction, hybrid fusion, metadata filters, LLM query rewriting, reranking, structural lookup and expansion (fixed rules and an agent), each with and without a score gate, on two chunkings. Qdrant is the only vector store, the JSONL corpus is loaded into memory, and every experiment record is a file under `results/`. The design is in `plan.md`.

## Setup

```bash
uv sync                                                    # .venv with all deps
uv run python -m ipykernel install --user --name ikincitik --display-name "ikincitik (uv)"
cp .env.example .env                                       # or add the new keys (below) to an existing .env
docker compose build encoder                               # model server image (once; CPU and GPU services share it)
docker compose run --rm --no-deps -e HF_HUB_OFFLINE=0 encoder python download_models.py   # jina-reranker-v3, once
uv run jupyter lab                                         # pick the "ikincitik (uv)" kernel
```

The retrieval experiments time-share the GPU in **two phases** (one swap; stop other vLLMs such as skyrag-vllm or aviation-vllm* first):

```bash
# phase A, notebooks 04-07: encoder on the GPU, vLLM STOPPED
docker compose stop vllm encoder && docker compose up -d qdrant && docker compose up -d --force-recreate encoder-gpu
# phase B, notebook 08: vLLM on the GPU (VLLM_CONFIG=serve.retrieval.yaml in .env), encoder on the CPU
docker compose stop encoder-gpu && docker compose up -d qdrant vllm && docker compose up -d --force-recreate encoder
```

### Services (`compose.yaml`)

| service | runs on | what |
|---|---|---|
| `vllm` | GPU (phase B) | Qwen3.8-27B-INT4 (`RedHatAI/Qwen3.8-27B-INT4`, rev `c063053e…`), weights from the external volume `aviation-vllm-cache`. Options come from `vllm/${VLLM_CONFIG}`. |
| `qdrant` | CPU | `qdrant/qdrant:v1.18.3-unprivileged`, volume `ikincitik_qdrant`, ports 6333 (REST) and 6334 (gRPC). |
| `encoder` | CPU, no GPU devices (phase B) | Built from `encoder/`: bge-m3, bge-reranker-v2-m3 and jina-reranker-v3, port 8081. |
| `encoder-gpu` | GPU (phase A; compose profile `gpu`) | The same image and API on the GPU, also port 8081, so only one encoder runs at a time. **Never together with vLLM.** |

**Two vLLM serve configs**, chosen with `VLLM_CONFIG` (in `.env` or on the command line; the default is `serve.yaml`):
- `vllm/serve.yaml`: page extraction (vision on, `max-model-len 12288`). It stays unchanged, so the extraction run remains reproducible.
- `vllm/serve.retrieval.yaml`: retrieval experiments (`language-model-only`, `max-model-len 32768`, `gpu-memory-utilization 0.80`). Its KV cache holds 43,690 tokens, 1.33 × a full 32,768-token request (notebook 08 checks the startup log), and an 18k-token prompt (the agent's worst case) runs. 0.85 left no headroom next to the desktop's ~3 GB on the same card, and the first request died with CUDA OOM. At 0.82 the memory profile varied between starts.
- N-gram speculative decoding is **off** in both: on this hybrid model it corrupted greedy output.

**Why two phases.** The rerankers run ~8× faster on the GPU. Measured per question with 50 base units, fp32: bge takes 5.7 s on the GPU and 33 s on the CPU; jina takes 14.4 s and 141 s. That is ~1 h instead of ~9 h for X1 + X2, with the same scores (identical rankings, |Δ| ≤ 4e-5). No run needs a reranker and the LLM at the same time. In phase B the encoder only embeds short texts (HyDE passages, agent search queries), which takes seconds on the CPU. The device of each session is recorded in every run's `config.json`.

**`.env` keys:** `VLLM_ENDPOINT`, `VLLM_API_KEY`, `VLLM_MODEL` (the vLLM container), `VLLM_CONFIG` (serve file), `ENCODER_URL` (default `http://127.0.0.1:8081`), `ENCODER_THREADS` (torch threads in the encoder, default 12 = physical cores), `QDRANT_URL` (default `http://127.0.0.1:6333`), `BGE_M3_TOKENIZER` (bge-m3 `tokenizer.json` for token counts).

### `encoder/`: the model container (CPU and GPU)

`python:3.12-slim` (pinned by digest) with the CUDA 12.6 build of torch 2.14.1 (it also runs on the CPU), transformers 5.22.1 and FlagEmbedding 1.4.2. `requirements.txt` is fully pinned and is compiled from `requirements.in` (the command is in its header). All three models work with one `transformers` version, so the service is not split. `ENCODER_DEVICE` (`cpu` / `cuda`) picks the device. Models run in fp32 on both devices; on the GPU, TF32 is off, so the numbers match the CPU closely. Models are loaded offline (`HF_HUB_OFFLINE=1`) from local snapshot paths, so each revision is pinned:
- bge-m3 (`5617a9f6…`) and bge-reranker-v2-m3 (`953dc6f6…`): `../etoeragcb/model-cache/huggingface/hub`, mounted read-only;
- jina-reranker-v3 (`d7d7e73b…`): the volume `ikincitik_models`, downloaded once with `download_models.py` (the hub is online only for that step). It loads with `trust_remote_code`, so the pinned revision also pins its model code. **Licence: CC BY-NC 4.0** (non-commercial; academic use is allowed).

Endpoints:
- `POST /embed` `{texts, dense, sparse, colbert}` → bge-m3 dense (1024-d, normalised), learned-sparse weights and ColBERT vectors (float16) from **one forward pass**, `max_length` 8192.
- `POST /rerank` `{model: "bge" | "jina", query, documents}` → one score per document, in input order. bge returns raw logits (no sigmoid) with `max_length` 8192 (FlagEmbedding's default of 512 would cut off about half the base units). jina is listwise, takes ≤ 64 documents per call, and has `max_doc_length` 8192 (its default is 2048).
- `POST /tokens` → token counts under a reranker's tokenizer (the truncation check).
- `GET /health` → loaded models, revisions, device, dtype and library versions. These are stamped into every run's `config.json`.

## Pipeline (notebooks/)

| notebook | does | writes |
|---|---|---|
| `00_vllm_smoke` | waits for vLLM, sends one page image | – |
| `01_segment_toc` | outline bookmarks → exact (page, y) section boundaries, coverage check | `data/sections.json` |
| `02_extract_vlm` | per page region: image + PDF text layer → Qwen → Markdown; scored, retried, cached | `data/vlm_cache/`, `data/flags.csv` |
| `03_assemble_chunks` | stitch pages, rebuild nesting, write outputs, boundary checks | `data/sections.jsonl`, `groups.jsonl`, `chunks.jsonl` |
| `04_enrich_window_embed` | engine_type, base-unit spans, cross-references; 500-token windows and their checks; bge-m3 (dense, sparse, ColBERT) and BM25 vectors for both indexes | `data/windows_500.jsonl`, `data/cross_refs.jsonl`, `data/embeddings/` |
| `05_ingest_qdrant` | creates both collections, upserts, ingestion checks (count, ids, round-trip, self-retrieval), HNSW vs exact, ColBERT engine agreement, server-side BM25 comparison | Qdrant volume |
| `06_load_questions` | validates the gold quotes, builds span-based qrels for both indexes | `data/eval/qrels.jsonl` |
| `07_run_retrievers` | **phase A** (GPU encoder, vLLM stopped): question embeddings, each signal's top-50, the grid without LLM (L1 … T2+T3), the rerankers X1/X2, and the ANN, RRF and gate checks | `data/embeddings/queries_questions.npz`, `data/cache/search.jsonl`, `data/cache/rerank_*.jsonl`, `results/runs/` |
| `08_llm_and_agent` | **phase B** (vLLM, CPU encoder): vLLM KV-cache check; HyDE passages (Q1), keyword expansions (Q2) and engine-type filters (F1); the Q1/Q2/F1 runs; the agent T4; reproducibility, label-leakage, one-hop and completeness checks | `data/cache/llm.jsonl`, `data/embeddings/queries_{hyde,agent}.npz`, `results/runs/` |
| `09_analysis` | metrics, bootstrap CIs, paired randomisation tests with Holm, tables, plots, metrics files, diagrams, output checks | `results/run_metrics.parquet`, `results/question_metrics.parquet`, `results/{units,win500}/metrics.{csv,xlsx}`, `results/granularity.csv`, `results/diagrams/` |

**Run order for the experiments:** phase A: 04 → 05 → 06 → 07 (`encoder-gpu`, **vLLM stopped**); swap; phase B: 08 (vLLM + CPU `encoder`); then 09 (no models). Approximate times: corpus embedding in 04 takes a few minutes on the GPU; the grid without LLM in 07 takes minutes; the rerankers in 07 take ~1 h on the GPU; the query-side LLM work in 08 takes ~10–25 min; the agent takes ~1–2 h. All of it is cached and resumable (a run continues from `results/runs/<run_id>/partial.jsonl`). For v2, run 07 and 08 again with `PARAMS_VERSION = "v2"`.

The code lives in `src/`; the notebooks only call it and expose the parameters.

## Code (src/)

**`csparse/`**: PDF → sections and chunks (`segment.py`: section boundaries from the outline; `vlm.py`: page-region extraction with Qwen; `merge.py`: stitching, nesting, chunking, `Corpus`).

**`csrag/`**: the retrieval experiments.

| module | what |
|---|---|
| `corpus.py` | in-memory corpus for both indexes (extends `csparse.merge.Corpus`): `units`, `units_of`, `refs_of`, `merge_spans`, `n_tokens` (bge-m3 tokenizer) |
| `windows.py` | 500/50 section-bounded splitter in bge-m3 tokens, with spans, checks and stats |
| `enrich.py` | `engine_type`, whitespace normalisation with position maps, base-unit spans, cross-reference extraction and normalisation, point ids, `corpus_version` |
| `embed.py` | encoder client (`/embed`, `/rerank`), fastembed BM25, the corpus embedding store and query-embedding caches |
| `index.py` | Qdrant collections: create, upsert, ingestion checks, ColBERT engine check |
| `runs.py` | caches (`llm`, `search`, `rerank_bge`, `rerank_jina`) and run records; `execute` runs one config, resumably |
| `gate.py` | the score gate (gate A on first stages, gate B after reranking) |
| `retrievers/base.py` | `Hit`, the fixed parameters (`PARAMS["v1"]`), method configs and the run grid |
| `retrievers/__init__.py` | `Pipeline` (interprets any config: query transform → first stages → gate A → fusion → filter → rerank → gate B → lookup → expansion) and `Resources` |
| `retrievers/qdrant_search.py` | cached Qdrant search per signal, and the client-side ColBERT MaxSim (`ColbertIndex`) |
| `retrievers/fusion.py` | client-side RRF (k = 60, ranks from 1) and Qdrant's server-side RRF for the check |
| `retrievers/query_llm.py` | vLLM client with a request-hash cache; HyDE, keyword expansion, filter extraction |
| `retrievers/rerank.py` | bge (pointwise, cached per query and unit) and jina (listwise, cached per query and ordered list) |
| `retrievers/structural.py` | T1 id lookup; the budgeted context builder; T0, T2, T3 and T2+T3 |
| `retrievers/agentic.py` | T4: the agent with `expand`, `follow_refs`, `search` and `done`, one JSON-schema LLM call per step |
| `evaluate.py` | question validation, span-based qrels, metrics (incl. evidence recall, gate diagnostics, supporting recall, all-gold), bootstrap CIs, randomisation tests, Holm |
| `report.py` | per-index `metrics.csv`/`metrics.xlsx`, `granularity.csv`, Mermaid pipeline diagrams |
| `qgen.py` | question set 2: `fill` (derived fields), `check` (schema, verbatim quotes, family rules, twin table), `pool` (candidates for the single-source judge: BM25 over sections plus structure), `stats`, `overlap`, `refs`; needs no services or tokenizer |

## Indexes (Qdrant)

Two collections over the same text, with the same vectors and payload schema; window size is the only difference (RQ7):

| collection | units | how it is built |
|---|---|---|
| `cse_units_v1` | 252 base units (`chunks.jsonl`) | one unit per CS-E or AMC section, split only if longer than 1500 tokens and only at its own `(a)/(b)/…` paragraphs |
| `cse_win500_v1` | 436 windows (`windows_500.jsonl`) | ≤ 500 bge-m3 tokens, ≥ 50-token overlap, never across a section; a section of ≤ 500 tokens is one window (116 of 175) |

Named vectors: `dense` (1024, Cosine), `colbert` (multivector, MAX_SIM, float16, `hnsw m=0`), `bge_sparse` (bge-m3 lexical weights) and `bm25` (fastembed, `modifier=IDF`, `avg_len` per index). The payload carries `section_id`, `group_id`, `engine_type`, `cross_refs`, the character spans and the text; there are keyword and integer payload indexes. Sections and groups are not indexed separately: section-level ranking dedupes hits by `section_id` (MaxP), and expansion fetches parent texts from the in-memory corpus.

**ColBERT and Qdrant's multivector limit.** Qdrant accepts at most 1,048,575 numbers per multivector, i.e. 1023 ColBERT vectors of 1024 dimensions. Every window fits, but 60 of the 252 base units are longer (54% of the corpus tokens). Those points are stored without their `colbert` vector. C1 and M1 compute ColBERT MaxSim client-side, exactly, over each unit's full cached vectors, for **both** indexes (`colbert_engine: "client"` in the run config). This is the same math as Qdrant's MAX_SIM with DOT: notebook 05 checks that the two engines agree (to within 4e-6) where Qdrant holds the vectors.

## Data (data/)

| file | what | written by | in git |
|---|---|---|---|
| `sections.json` | section boundaries (page, y) from the PDF outline | 01 | yes |
| `vlm_cache/` | per page-region Qwen output | 02 | no (gitignored; regenerable with vLLM) |
| `flags.csv` | retried, `ok_fig` and fallback regions | 02 | yes |
| `sections.jsonl` | full text of each section (175) | 03 | yes |
| `groups.jsonl` | each CS-E with all its AMCs (92) | 03 | yes |
| `chunks.jsonl` | base units (252), the canonical text of `cse_units_v1` | 03 | yes |
| `windows_500.jsonl` | 500-token windows, the canonical text of `cse_win500_v1` | 04 | yes (deterministic) |
| `cross_refs.jsonl` | `index, src_unit_id, src_section_id, dst_section_id, ref_text, order` | 04 | yes (deterministic) |
| `eval/questions.jsonl` | 100 questions (92 answerable, 8 unanswerable); see `eval/README.md` | – | yes |
| `eval/qrels.jsonl` | span-based qrels: unit level per index, section and group level | 06 | yes (deterministic) |
| `eval/specs/` | generation specs for question set 2 (four families, all answerable): rules, record format, judge checks, freeze procedure | – | yes |
| `eval/set2/` | question set 2, one file per family, plus `pools/` for the single-source check; written from `eval/specs/`, checked with `csrag.qgen` | – | yes |
| `embeddings/` | `<collection>.{npz,jsonl}` corpus vectors and `queries_*.npz` query vectors | 04, 08 | no (gitignored, large; regenerable with the encoder) |
| `cache/` | `llm.jsonl` (every LLM output), `search.jsonl` (each signal's top-50), `rerank_bge.jsonl`, `rerank_jina.jsonl` | 07, 08 | yes (the LLM outputs make reruns exact) |

Record formats:
- **chunks.jsonl**: `chunk_id`, `section_id` (`CS-E 20`, `AMC E 20(f)`), `group_id` (`CS-E 20`), `kind` (`cs` | `amc` | `appendix` | `amc_general`), `subpart`, `breadcrumb`, `part_index/part_count`, `page_start/end`, `n_tokens` (cl100k), `body`, `text` (= one header line + body). A section is split only if it is longer than `MAX_TOKENS` (default 1500), and only at its own `(a)/(b)/…` paragraphs or sub-headings.
- **windows_500.jsonl**: `window_id` (`<section_id>#w<n>`), `section_id`, `group_id`, `kind`, `subpart`, `engine_type`, `part_index`, `part_count`, `char_start`, `char_end` (span in the section's whitespace-normalised body), `page_start`, `page_end`, `breadcrumb`, `n_tokens_bge` (body only), `body`, `text` (header line + body).
- **sections.jsonl**: the full, unsplit text of each section. **groups.jsonl**: one record per CS-E, with the CS text followed by all of its AMCs. `csparse.merge.Corpus(...).expand(chunk_id, to="group" | "section")` expands a retrieval hit; `csrag.Corpus` adds the windows, spans and cross-references.

## Results (results/)

| path | what |
|---|---|
| `runs/<run_id>/` | one run: `config.json` (full config, `config_hash`, `params_version`, `corpus_version`, model revisions), `hits.parquet` (`question_id, rank, unit_id, section_id, score, stage, hop, n_tokens, in_context, source`), `stats.jsonl` (latency, LLM tokens, context tokens, gate trace per question), `trace.jsonl` (T4 agent steps) |
| `run_metrics.parquet` | `run_id, index, level, metric, value, subset, n, ci_lo, ci_hi` for every run |
| `question_metrics.parquet` | the same metrics per question (input to the paired tests) |
| `units/metrics.csv`, `win500/metrics.csv` | one row per run of that index: config, readable stages, every `level × metric`, the section nDCG@10 CI, gate diagnostics, multi-section metrics with their n, cost |
| `units/metrics.xlsx`, `win500/metrics.xlsx` | the same table plus `by_question_type`, `by_engine_type`, `by_kind`, `mentions_section_id`, `length_bucket`, `significance`, `gate`, `per_question` and `unanswerable` sheets |
| `granularity.csv` | RQ7: windows vs units (gate off), section level and evidence recall, Holm-corrected |
| `diagrams/<run_id>.md` | Mermaid flowchart of a run generated from its config, with its metric panel |

Run ids read `<method>[-b<budget>]-<index>-gate<on|off>-<params_version>-<hash8>`. To regenerate: notebooks 07 and 08 (runs; they skip completed ones) and notebook 09 (all files under `results/` except `runs/`). Scored metrics use the 92 answerable questions; the 8 unanswerable ones are only listed in the `unanswerable` sheets. Tables are sorted by section nDCG@10 for readability only: no run is selected as a winner.

**Parameters.** `PARAMS["v1"]` in `src/csrag/retrievers/base.py` is fixed before any run. After reading all v1 metrics, the parameters are revised once, by hand, as `PARAMS["v2"]`, and the grid runs again (notebooks 07 and 08 with `PARAMS_VERSION = "v2"`). Both versions are kept and reported.

## Notes

- The VLM must reproduce the text layer's words exactly (word recall and precision ≥ 0.995, no missing numbers, well-formed Markdown). Otherwise it retries once, then falls back to the text layer without table structure. Status `ok_fig` means the output drops only text drawn inside a vector figure. Retried, `ok_fig` and fallback regions are listed in `data/flags.csv`. `rescore_cache()` re-applies changed scoring rules to the cache without calling the model.
- The scoring is bag-of-words. It catches dropped and invented words, not a word placed in the wrong table cell. Use `show()` in notebook 02 to compare against the page image.
- Nesting is rebuilt from the markers, not taken from the model. CS-E's 5-level scheme (a)(1)(i)(A)(1) and the (i)/(v)/(x) letter-or-roman ambiguity are handled in `merge.indent_markers`.
- BM25 tokenisation splits "CS-E 740(c)(4)" into `cs`, `e`, `740`, `c`, `4`; the T1 id lookup covers id-heavy questions. Qdrant's server-side BM25 (`models.Document(..., model="qdrant/bm25")`) gives the same token ids and weights as fastembed on v1.18.3 (checked in notebook 05); fastembed is the method of record.
