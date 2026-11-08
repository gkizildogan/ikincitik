# Question set 2: generation specs

Set 2 is a second question set for the retrieval experiments, written by Claude from these specs. v1 (`../questions.jsonl`) cannot separate the methods well:

- its questions reuse the gold section's wording (84% of a question's content stems occur in its gold section on average, measured with `qgen overlap`);
- they name the engine type outright;
- only 6 name a section id;
- none needs more than one section.

Set 2 is written in four families. Each has its own spec; this file holds what they share. **Every set-2 question is answerable.**

| family | file | spec | questions | what it measures |
|---|---|---|---|---|
| paraphrase | `set2/paraphrase.jsonl` | `paraphrase.md` | 184 (2 per answerable v1 question) | sensitivity to wording with the gold held fixed: BM25 vs dense and ColBERT, HyDE (Q1), keyword expansion (Q2), rerankers (RQ1, RQ3, RQ4) |
| section_id | `set2/section_id.jsonl` | `section_id.md` | 50 | id lookup (T1), and how BM25 and dense handle id queries (RQ5) |
| twin_topic | `set2/twin_topic.jsonl` | `twin_topic.md` | 60 (30 piston/turbine pairs) | engine-type filtering: F1 vs F0 vs H1, BM25 vs dense on near-identical texts (RQ5) |
| multi_section | `set2/multi_section.jsonl` | `multi_section.md` | 90 | expansion T2, T3, T2+T3 and the agent T4; supporting recall and all-gold metrics (RQ6) |

That is 384 questions in total.

## Rules for every family

1. **Evidence first, question last.** Pick the gold section(s), copy the quote(s), write the reference answer, then write the question.
2. **Quotes are verbatim.**
   - A `gold.quote` is copied character for character from the section's `body` in `data/sections.jsonl`. Don't copy it from `text` or from the PDF.
   - It occurs once in that body.
   - It is the sentence, or the two sentences, that carry the answer (aim for 60–300 characters).
   - `qgen check` rejects anything else.
3. **One answer location.**
   - Only the gold section(s) answer the question. If another section also answers it fully, make the question more specific or drop it.
   - Don't add the other section as an alternative gold. The metrics treat every gold quote as needed: evidence recall is the share of gold quotes covered.
   - In multi_section, every gold section is needed and together they are enough.
4. **Answerable only.** `answerable: true` everywhere. No questions about things CS-E does not cover.
5. **Applicability (CS-E 10(c), (d)).** Subparts A, B and C apply to piston engines; Subparts A, D, E and F apply to turbine engines. Subpart A applies to both. A Subpart D/E/F section never answers a piston question, and a Subpart B/C section never answers a turbine question.
6. **What you may read:**
   - `data/sections.jsonl`, `data/groups.jsonl`, `data/cross_refs.jsonl` and `data/windows_500.jsonl`;
   - `data/eval/questions.jsonl` (v1);
   - these specs and `src/csrag/qgen.py`.

   **Do not open** `results/`, `project_v1_output.md`, `data/cache/` or notebook outputs while generating. The targets come from the specs, never from which questions a method got wrong: picking questions by retriever outcome would build the result into the set.
7. **Do not run retrievers** on set-2 questions until the set is frozen: no Qdrant, encoder, vLLM, or notebooks 05–09. The pool for check B (`qgen pool`) uses its own BM25 over whole sections and the document structure, not the experiment's retrievers.
8. **Change nothing outside `data/eval/set2/`.** In particular, never edit v1 (`questions.jsonl`, `qrels.jsonl`).
9. **No outside content.** The question asks only for what the gold says. Neither the question nor the answer brings in facts from other documents or from general knowledge.
10. **Section ids appear only in the section_id family.** The one exception is the paraphrases of the 6 v1 questions that already name one. No other question names a section id, in any spelling.

## Record format

Each line of a family file is one question in the v1 format (`../README.md`, "Fields"), plus the fields below. `qgen fill` writes the fields marked *derived*; you write the rest.

| field | who | value |
|---|---|---|
| `question_id` | you | family prefix, see the family spec: `pa071h`, `id017`, `tw07p`, `ms045` |
| `question` | you | |
| `question_type` | you | the v1 types (`factual`, `numeric`, `list`, `procedural`, `definition`, `rationale`); `summary` is new (section_id only) |
| `answer`, `answer_aliases`, `answer_references` | you | as in v1 |
| `answerable` | derived | `true` |
| `gold` | you | `[{section_id, group_id, quote, role}]`, with role `primary` or `supporting` |
| `gold_kind`, `engine_type`, `subpart` | derived | from the gold sections, with v1's meaning: `engine_type` belongs to the gold section, and is `all` for Subpart A or when the gold has both types |
| `requires_all_gold` | you | `true` only where the family spec says so |
| `mentions_section_id` | you | `true` if and only if the question names a section id in any spelling (`qgen check` detects them) |
| `difficulty` | you | `easy`, `medium` or `hard`, your judgement |
| `derived_from`, `distractor_section_ids` | derived | `null` and `[]` (v1 used them only for unanswerable questions) |
| `source`, `corpus_version`, `review` | derived | `claude-generated`; the sha256 of `groups.jsonl`; `{status: unreviewed}` |
| `generator` | you | the model id of the session that wrote the question (or pass `qgen fill --generator <id>`) |
| `family` | you | `paraphrase`, `section_id`, `twin_topic` or `multi_section` |
| `asks_about_engine_type` | you | the engine type the *question* is about, judged from its wording: `piston`, `turbine`, `both` or `unspecified`. Unlike `engine_type`, it describes the question, not the gold section |
| `lexical_overlap` | derived | share of the question's distinct content stems that occur in the gold section(s) |
| `verification` | derived, then you | `{answer_from_gold, single_source, same_meaning, each_alone_fails, second_judge, notes}`. Each check is `pass`, `fail`, `pending` or `n/a`; `notes` is free text |
| family fields | you | see each spec |

## Tools

`src/csrag/qgen.py`. Run it from the repo root, after one `uv sync`. It needs no services and no bge-m3 tokenizer.

```bash
uv run python -m csrag.qgen overlap "draft question" "CS-E 750"   # overlap, shared and unshared stems, ids found
uv run python -m csrag.qgen fill data/eval/set2/<family>.jsonl     # derived fields, in place
uv run python -m csrag.qgen check data/eval/set2/<family>.jsonl    # schema, quotes, family rules; exit code 1 on errors
uv run python -m csrag.qgen pool data/eval/set2/<family>.jsonl     # candidate sections for check B
uv run python -m csrag.qgen stats data/eval/set2/*.jsonl           # counts per field, overlap, check status
uv run python -m csrag.qgen refs --section "CS-E 740"              # cross-group references (multi_section)
uv run python -m csrag.qgen twins                                  # the twin table (twin_topic)
```

`check --final` also requires every verification check of every record to pass. `check` validates whatever files you pass together, so check a twin pair or the whole family in one call.

## Workflow per batch

A batch is 10–25 records of one family. Each family spec lists its batches.

1. Take the batch's targets from the family spec, in the order given.
2. Read each gold section in full (`body` in `data/sections.jsonl`). For twin_topic and multi_section, read every section involved.
3. Write each record in order: gold quote(s), then the answer, then the question. Run `qgen overlap` while drafting when the family has an overlap cap.
4. Run `qgen fill`, then `qgen check`, until there are no errors.
5. Run the judge checks below: A and B for every record, plus C for multi_section and D for paraphrase. Record each result in `verification`, and add a line to `verification.notes` when you fixed something.
6. If a record fails, rewrite it and run the checks again. After two failed rewrites, delete it and take the next target. Paraphrase has no next target: write that paraphrase again from scratch.
7. Run `qgen check` one more time, then commit the batch and push.

## Judge checks

Each check is run by a **fresh subagent** that did not write the question. The subagent sees exactly the inputs listed: never the reference answer, never which section is gold. One subagent can judge a whole batch, question by question. If the judging model differs from the generator, record it in `verification.notes`.

### A. `answer_from_gold`

**Input:** the question, and the full `text` of the gold section(s).

> Below are one or more sections of EASA CS-E Amendment 9 and a question. Using only these sections, answer the question, then copy verbatim the sentences you used. If the sections do not fully answer the question, say so and say what is missing. Applicability: Subparts A, B and C apply to piston engines; Subparts A, D, E and F apply to turbine engines.
> Reply as JSON: `{"fully_answered": true|false, "answer": "...", "sentences": ["..."], "missing": "..."}`

**Pass** if all three hold:
- `fully_answered` is true;
- the judge's answer states the same facts as the reference `answer` (same numbers, conditions and list items, and nothing that contradicts it);
- the judge's sentences include or overlap the gold quote(s).

You compare the answers yourself. When unsure, ask another fresh subagent: "Do answers X and Y state the same facts? List every difference."

### B. `single_source`

**Input:** the question, and the full `text` of every section in its pool. `qgen pool` writes the pools to `data/eval/set2/pools/<family>.jsonl`. Each pool lists, in document order and unmarked, the gold section(s) and these candidates:
- BM25's top 6 sections;
- every section of the gold's group;
- the sections most linked to the gold by cross-references;
- any section named in the question;
- for twin_topic, both sides of the twin.

> Below are a question and N sections of EASA CS-E Amendment 9. Applicability: Subparts A, B and C apply to piston engines; Subparts A, D, E and F apply to turbine engines. For each section, label it FULL if that section alone answers the question completely, for the engine type and situation the question describes; PARTIAL if it is relevant but does not answer the question alone; NO otherwise. A section that would answer the question only for the other engine type is NO.
> Reply as JSON: `{"<section_id>": {"label": "FULL|PARTIAL|NO", "reason": "..."}, ...}`

**Pass:**
- section_id, paraphrase, twin_topic: the gold section is FULL and no other section is FULL;
- multi_section: no section outside the gold is FULL, and every gold section is FULL or PARTIAL.

A non-gold FULL means the question has a second answer location: make it more specific, or drop it. A pool is typically 8–16 sections and 10–25k tokens; pools that include the long sections (CS-E 740, AMC E 515, AMC E 780) are larger. Judge a large pool in two halves, with one subagent labelling each half.

### C. `each_alone_fails` (multi_section only)

Run check A once per gold section, giving the judge that section alone. **Pass** if every run returns `fully_answered: false`. This confirms the question needs all its sections.

### D. `same_meaning` (paraphrase only)

**Input:** the v1 original and the paraphrase, without any section text.

> Here are two questions about EASA CS-E Amendment 9. Would every correct and complete answer to question 1 also be a correct and complete answer to question 2, and the other way round? Check the engine type, aircraft type, conditions, exceptions, numbers and what exactly is asked for.
> Reply as JSON: `{"same": true|false, "difference": "..."}`

**Pass** if `same` is true. Check A also catches drift: a paraphrase that asks for something the gold does not say fails A.

### `second_judge` (optional, on your machine)

Checks A and B again, with a model from another family, such as Qwen through vLLM in phase B (`vllm/serve.retrieval.yaml`). Record `pass` or `fail`, or leave `pending` if you don't run it. A `fail` blocks `--final`.

Use Qwen only as a judge, never to write or rewrite questions. Qwen also writes the HyDE passages and keyword expansions and runs the filter and the agent, so questions in its own phrasing could favour those methods.

## Freezing

1. `qgen check --final data/eval/set2/*.jsonl` passes.
2. The owner reviews a seeded random sample of at least 10% of each family, sets `review.status` to `ok`, `fixed` or `rejected`, and deletes rejected records. No domain expertise is needed: check that the quote answers the question, and that the answer says what the quote says.
3. Commit the set as "Freeze question set 2". After that, records are not edited. A correction becomes a new record with a new id, and the analysis lists it.
4. Only now build qrels and run retrievers on set 2. Notebooks 06–09 read the v1 paths only. Wiring set 2 in, with its own qrels file and results root, is a separate change: `runs.execute` skips a completed run, so new questions cannot simply be appended to v1.

## Asking Claude for a batch

> Read `data/eval/specs/README.md` and `data/eval/specs/<family>.md`. Generate batch <n> of the <family> family into `data/eval/set2/<family>.jsonl`, following the workflow, including the judge checks with fresh subagents. Run `qgen check`, then commit and push.

Run the families in any order. paraphrase is the most mechanical, and multi_section is the most work per question.
