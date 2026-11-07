# Adding and updating documents in the stores

This runbook is for a coding agent. It explains how to add new regulation documents (e.g. CS-27, CS-25) next to CS-E Amendment 9, and how to update a document that is already stored (a new amendment, a fixed chunk). It covers the canonical JSONL corpus, the embedding stores and the Qdrant collections.

**Context:** today the repo holds one document, CS-E Amendment 9, used for a PhD retrieval experiment (`plan.md`, notebooks 04–09). The multi-document setup described here is meant for **production**. The two must not interfere.

---

## 0. Ground rules

1. **Never write to the experiment collections** `cse_units_v1` and `cse_win500_v1` (`COLLECTIONS` in `src/csrag/corpus.py`). Don't touch their unit files (`data/chunks.jsonl`, `data/windows_500.jsonl`), embedding stores (`data/embeddings/cse_*`) or `results/` either. The questions, qrels, distractor sections and every run record are tied to this exact corpus through `corpus_version`, the sha256 of the unit file. Production uses **new collection names** and its own data directory.
2. **Production collections are versioned and sit behind an alias.** Build `<name>_v<N+1>` next to the live `<name>_v<N>`, verify it, then switch the alias in one atomic call. Never mutate the live collection in place (see section 5).
3. **Unit ids must be unique across all documents.** Fail hard on a collision; never let one document overwrite another (section 2.2).
4. **Any change to the set of documents means re-embedding BM25 for every unit and re-upserting every point** (section 2.1). Only the new texts are re-encoded with bge-m3.

---

## 1. The three Qdrant operations (`src/csrag/index.py`)

| operation | what it does | when to use it |
|---|---|---|
| `create_collection(client, index, recreate=False)` | Creates an **empty** collection with the schema: named vectors `dense`, `colbert`, `bge_sparse`, `bm25` (IDF modifier), plus keyword and integer payload indexes. **If the collection exists, it does nothing.** | Always safe to call. |
| `create_collection(..., recreate=True)` | **Deletes** the collection and all its points, then creates it empty. | A brand-new versioned collection, or a schema change. Never on a live or experiment collection. |
| `upsert(client, index, units, store, corpus_version)` | Insert-or-overwrite. Point id = `uuid5(unit_id)` (`enrich.point_id`), so the same `unit_id` overwrites its point and a new `unit_id` adds a point. **It never deletes.** | Filling a new collection; re-running is idempotent. |

Consequence: a unit removed from the unit file (a deleted section, a renamed id, a re-split chunk) **stays in Qdrant as a stale point** after an upsert. `index.verify` reports it under `extra_in_qdrant` and sets `ok = False`. Removing stale points needs either a rebuild into a fresh collection (preferred) or an explicit `client.delete(...)` by point id.

---

## 2. Why "just upsert the new document's chunks" is wrong

### 2.1 BM25 depends on the whole collection
- **Document side:** fastembed's BM25 (`embed.BM25`) saturates term frequencies using `avg_len`, the mean unit length of the index. Notebook 04 computes it with `embed.bm25_avg_len` and stores it in the embedding store's `meta["bm25"]`. A new document changes `avg_len`, so every existing BM25 vector becomes stale.
- **Server side:** the `bm25` vector uses `modifier=IDF`, so Qdrant computes the IDF over all points in the collection. Adding a document changes the BM25 score of every query, even for points you didn't touch.
- **Handled already:** `EmbeddingStore.build` compares the stored BM25 params with the current ones. If they differ, it recomputes BM25 for **all** units and drops rows that are no longer in the unit list. It re-encodes with bge-m3 only the units whose `text_sha` is new or changed.
- **So:** rebuild the store over the full combined unit list, then upsert **all** points.

The bge-m3 vectors (`dense`, `bge_sparse`, `colbert`) depend only on a unit's own text, so they can be reused across documents and versions.

### 2.2 Point ids come only from the `unit_id`
`point_id(unit_id) = uuid5(POINT_NAMESPACE, unit_id)`. If a new document produces a `unit_id` that already exists, for example a shared "AMC General" or an appendix id, the upsert **silently overwrites** the other document's point. The in-memory `Corpus` dicts are keyed by `unit_id` too, so the collision also corrupts lookups. Section 3.2 says how to prevent this.

### 2.3 The metadata is CS-E-specific
`engine_type` comes from the CS-E subparts (`enrich.ENGINE_TYPE`: A → all, B/C → piston, D/E/F → turbine). `Corpus._base_units` calls `engine_type(c["subpart"])`, which raises `KeyError` on any other document's subparts.

---

## 3. One-time code changes before the second document

The pipeline is hard-wired to CS-E in these places. Make them per-document, keyed by a `doc_id` such as `"CS-E"` or `"CS-27"`, without changing the CS-E results: the experiment must still reproduce byte-identical `data/*.jsonl`.

### 3.1 Extraction (`src/csparse/`, notebooks 01–03)

| place | CS-E-specific part |
|---|---|
| `segment._ID_RE`, `segment.build_sections` (`kind = "cs" if prefix == "CS-E"`) | Section-id grammar: `CS-E 20`, `AMC E 20(f)`, `AMC to CS-E`, appendices |
| `segment._SUBPART_RE`, `segment.FIRST_BODY_PAGE = 16` | Subpart headings and where the body starts after the ToC and preamble |
| `merge.HEADER_LEAK_RE` | Header and footer text: `Page \d+ of 268`, `CS-E — Amendment 9` |
| `vlm.py` system prompt | "You convert one region of a page from EASA CS-E…" |
| the PDF itself | The CS-E PDF was pre-cropped to remove its header and footer. Do the same for each new PDF, or add a per-document crop box. |

Put these in a per-document profile, e.g. a dataclass with the id regex, subpart regex, first body page, header-leak regex, prompt wording and crop box. The CS-E profile must keep today's values exactly. Notebook 03's `check_outputs` must give all zeros for every document.

Keep the acceptance rule for VLM output unchanged: word recall and precision ≥ 0.995 against the text layer, and no missing numbers; otherwise retry once, then fall back to the text layer. Page extraction needs vLLM with `VLLM_CONFIG=serve.yaml` (vision on). The retrieval config `serve.retrieval.yaml` is text-only.

### 3.2 Ids and payload (`src/csrag/`)
- Add a `doc_id` field to every section, group, chunk and window record, and to `index.PAYLOAD_FIELDS`. Add it to `index.KEYWORD_INDEXES` too, so you can filter by document.
- Make the point id document-scoped: `point_id(f"{doc_id}:{unit_id}")`. Keep this **only** for the production collections. The experiment collections keep `uuid5(unit_id)`.
- When the combined corpus is built, **assert** that every `section_id`, `group_id`, `chunk_id` and `window_id` is unique across documents. If real ids collide, prefix the readable id with the document. Never let the collision through.
- Replace `engine_type` with a per-document metadata mapping, e.g. engine type for CS-E and rotorcraft category for CS-27. Store `null` where a field doesn't apply rather than inventing a value.
- `corpus_version`: stamp the combined unit file's sha256, as now, and add `doc_versions`, a `{doc_id: sha256 of that document's unit file}` map, so you can tell which document changed.

### 3.3 Cross-references (`enrich.REF_RE`, `normalize_ref`, `_amc_lookup`, `extract_refs`)
`REF_RE` only matches `CS-E N`, `AMC E N` and `AMC to CS-E N`. Each document needs its own reference patterns. References may also cross documents (CS-27 text citing CS-E). Resolve a reference only to a section that exists in the combined corpus, as today, and record the target's `doc_id`. The T1 id lookup (`structural.id_lookup`, which uses `corpus.section_ids`) works across documents automatically once the ids are unique.

### 3.4 Corpus and collection names (`src/csrag/corpus.py`)
`COLLECTIONS` and `UNIT_FILES` are module constants pointing at the experiment. Pass the production names and a production data directory explicitly instead. Don't edit the constants: notebooks 04–09 depend on them.

### 3.5 LLM prompts (`retrievers/query_llm.py`, `retrievers/agentic.py`)
The HyDE, keyword, filter and agent prompts say "EASA CS-E Amendment 9" and "the CS-E section with all its AMCs". For production, make the wording cover all documents. Prompt hashes are part of `config_hash`, so this doesn't affect the experiment.

### 3.6 Suggested production layout
```
prod/
  docs/<doc_id>/            source PDF, sections.json, vlm_cache/, flags.csv,
                            sections.jsonl, groups.jsonl, chunks.jsonl   (per document, from 01-03)
  corpus/                   combined chunks.jsonl, windows_500.jsonl, cross_refs.jsonl   (built from docs/*)
  embeddings/               one EmbeddingStore per index
```

Name the embedding store after the **index**, e.g. `regs_units`, not after the versioned collection. `EmbeddingStore` caches by `text_sha` within one store file, so a stable store name means a new collection version re-encodes only new or changed texts. A store named after `regs_units_v3` would start empty and re-encode everything.

---

## 4. Procedure: add a new document

GPU phases: page extraction needs vLLM (`serve.yaml`). Embedding needs the encoder, preferably `encoder-gpu` with vLLM **stopped**, because the two never fit on the card together. See `CLAUDE.md` for the swap commands.

1. **Extract** the new PDF with notebooks 01–03 and its document profile into `prod/docs/<doc_id>/`. Check:
   - `coverage_report`: no orphans or overlaps;
   - `check_outputs`: all zeros;
   - `flags.csv` reviewed with `show()` in notebook 02.
2. **Combine.** Concatenate all documents' records in a fixed document order, then fixed section order. Run the uniqueness assertion (section 3.2). Build the windows (`windows.py`, 500/50, never across a section) and the cross-references over the **combined** corpus.
3. **Embed** with `EmbeddingStore.build` over the full combined unit list, for each index. Expect `bm25_recomputed: True` and `embedded` ≈ the new document's unit count. Check that no unit exceeds bge-m3's 8192-token input (`n_tokens_bge`); the encoder truncates longer input silently.
4. **Build a new collection version.** Call `create_collection(..., recreate=True)` on `<name>_v<N+1>` (a new name, so recreate deletes nothing live). Then `upsert` **all** units with the new `corpus_version`.
5. **Verify** (section 6). Don't continue if any check fails.
6. **Switch the alias** atomically. In one `client.update_collection_aliases(...)` call, delete the alias from `<name>_v<N>` and create it on `<name>_v<N+1>`. Clients always query the alias, never a versioned name.
7. **Keep `<name>_v<N>`** until the new version has served traffic without problems, as the rollback target. Then delete it.

## 5. Procedure: update a stored document (new amendment, corrected text)

Use the same steps as section 4, with step 1 replaced by re-extracting only the changed document into its `prod/docs/<doc_id>/`. Rebuild into a new collection version even for a small change:

- The corpus is small (~155k bge-m3 tokens per document), and unchanged texts hit the embedding cache, so a rebuild is cheap.
- Removed or renamed sections disappear automatically: a fresh collection has no stale points, and `EmbeddingStore.build` prunes rows that aren't in the unit list when BM25 is recomputed.
- `avg_len` and the IDF almost always change with an amendment, so every point needs re-upserting anyway.

**In-place upsert into the live collection is acceptable only** when all of these hold:
- the set of `unit_id`s is unchanged;
- only some texts changed;
- the changes are too small to move `avg_len`. Check: `bm25_recomputed` must be `False`. If it's `True`, do a full rebuild.

Even then, run `verify` afterwards. Never use `recreate=True` on a collection behind an alias: clients would see an empty collection during the rebuild.

## 6. Verification before switching the alias

Run each check for every index:
- `index.verify(...)["ok"]` is true:
  - point count equals the number of units;
  - no `missing_in_qdrant` and no `extra_in_qdrant`;
  - vectors round-trip exactly (dense within 1e-6);
  - self-retrieval at rank 1 is ≥ 0.99 for dense and BM25.
- `index.check_colbert_engines` agrees, within 1e-3, on points that hold a ColBERT vector. Qdrant caps a multivector at 1023 vectors (`index.MAX_MULTIVECTOR`), so longer units are stored without one, and ColBERT scoring must use the client-side `ColbertIndex` over the cached vectors.
- HNSW and exact dense search give identical top-50 lists for a sample of queries (notebook 05's check).
- Every point carries `doc_id`, `corpus_version` and `doc_versions`, and the per-document point counts match the per-document unit files.
- Regression: a fixed set of known questions per document still returns its gold section in the top-k. For CS-E, `data/eval/questions.jsonl` with `data/eval/qrels.jsonl` can serve as a smoke test against the production collection.

## 7. Rollback

Point the alias back at `<name>_v<N>` with one `update_collection_aliases` call. Nothing else is needed, because versions are never mutated after the switch. For extra safety, take a Qdrant snapshot (`client.create_snapshot(<collection>)`) of the live collection before each switch.
