# CS-E Amendment 9 → section-bounded Markdown for RAG

Turns `cs-e_amendment_9.pdf` (EASA CS-E Amendment 9, with the header and footer cropped off) into retrieval units. There is one unit per `CS-E X` and one per `AMC E X…`, with lists, glossaries and tables kept intact across page breaks. `[Amdt No: …]` notes are removed and figures are skipped.

## Setup

```bash
uv sync                                                    # .venv with all deps
uv run python -m ipykernel install --user --name ikincitik --display-name "ikincitik (uv)"
cp .env.example .env
docker compose up -d     # vLLM + Qwen3.8-27B-INT4 with vision; stop other vLLMs on the GPU first
uv run jupyter lab       # pick the "ikincitik (uv)" kernel
```

## Pipeline (notebooks/)

| notebook | does | writes |
|---|---|---|
| `00_vllm_smoke` | waits for vLLM, sends one page image | – |
| `01_segment_toc` | outline bookmarks → exact (page, y) section boundaries, coverage check | `data/sections.json` |
| `02_extract_vlm` | per page region: image + PDF text layer → Qwen → Markdown; scored, retried, cached | `data/vlm_cache/`, `data/flags.csv` |
| `03_assemble_chunks` | stitch pages, rebuild nesting, write outputs, boundary checks | `data/sections.jsonl`, `groups.jsonl`, `chunks.jsonl` |

The code lives in `src/csparse/` (`segment.py`, `vlm.py`, `merge.py`); the notebooks only call it and expose the parameters.

## Output records

- **chunks.jsonl**: `chunk_id`, `section_id` (`CS-E 20`, `AMC E 20(f)`), `group_id` (`CS-E 20`), `kind` (`cs` | `amc` | `appendix` | `amc_general`), `subpart`, `breadcrumb`, `part_index/part_count`, `page_start/end`, `n_tokens`, `body`, `text` (= one header line + body). A section is split only if it is longer than `MAX_TOKENS` (default 1500), and only at its own `(a)/(b)/…` paragraphs or sub-headings.
- **sections.jsonl**: the full, unsplit text of each section.
- **groups.jsonl**: one record per CS-E, with the CS text followed by all of its AMCs. `csparse.merge.Corpus(...).expand(chunk_id, to="group" | "section")` expands a retrieval hit.

## Notes

- vLLM options are in `vllm/serve.yaml` (mounted into the container). N-gram speculative decoding is **off**: on this hybrid model it corrupted greedy output.
- The VLM must reproduce the text layer's words exactly (word recall and precision ≥ 0.995, no missing numbers, well-formed Markdown). Otherwise it retries once, then falls back to the text layer without table structure. Status `ok_fig` means the output drops only text drawn inside a vector figure. Retried, `ok_fig` and fallback regions are listed in `data/flags.csv`. `rescore_cache()` re-applies changed scoring rules to the cache without calling the model.
- The scoring is bag-of-words. It catches dropped and invented words, not a word placed in the wrong table cell. Use `show()` in notebook 02 to compare against the page image.
- Nesting is rebuilt from the markers, not taken from the model. CS-E's 5-level scheme (a)(1)(i)(A)(1) and the (i)/(v)/(x) letter-or-roman ambiguity are handled in `merge.indent_markers`.
