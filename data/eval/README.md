# CS-E question set (`questions.jsonl`)

100 single-turn questions for retrieval evaluation: **92 answerable** and **8 unanswerable** from the corpus.

## How it was made
- **Generator:** Claude (`claude-opus-5-5`), reading only `data/groups.jsonl`. The file's sha256 is stored in `corpus_version`.
- **Answerable (`q001`–`q092`):** one question per group.
  - The gold section is one section of the group: the CS, or one of its AMCs, alternating where a group has both. Result: 59 CS, 32 AMC, 1 AMC General.
  - The question is written from a text window of that section, taken at a seeded random offset so not every question comes from a section's opening paragraph.
  - The questions are self-contained: they name the engine type or context so that only one section answers them.
  - Only 6 questions name their own section id (`mentions_section_id`); no question names any other section.
- **Unanswerable (`q093`–`q100`):** near-miss rewrites of 8 answerable questions (`derived_from`).
  - Each keeps the vocabulary and topic but asks about something CS-E does not cover: hydraulic reservoirs, APUs, rotor brakes, a numeric ETOPS IFSD rate, SAF, an unlock time limit, hybrid-electric engines, a glycol coolant temperature.
  - Absence was checked against `groups.jsonl` by keyword search.
  - `distractor_section_ids` lists the sections a retriever is likely to return.
- **Automatic checks:**
  - every `gold.quote` is a verbatim substring of its gold section and of no other section;
  - ids and questions are unique;
  - the `mentions_section_id` flag matches the question text.
- **Human review:** pending. Every `review.status` is `unreviewed`.

## Fields
| field | meaning |
|---|---|
| `question_id`, `question` | id and question text |
| `question_type` | `factual`, `numeric`, `list`, `procedural`, `definition`, `rationale`, `unanswerable` |
| `answer` | reference answer (free text); for unanswerable questions, why the corpus does not answer it |
| `answer_aliases` | short acceptable forms, e.g. `"10%"`, `"3 minutes"` |
| `answer_references` | other sections or documents cited in the answer (e.g. `CS-E 510`). An answer that points the user to these sections, instead of quoting their content, is acceptable: following the reference is a separate query. |
| `answerable` | `true` / `false` |
| `gold` | `[{section_id, group_id, quote, role}]`; empty for unanswerable questions |
| `gold_kind` | `cs`, `amc` or `amc_general` |
| `requires_all_gold` | always `false` (single-section questions; no multi-hop) |
| `mentions_section_id` | the question names its gold section id (report metrics with and without these) |
| `engine_type`, `subpart` | `all` (Subpart A), `piston` (B, C) or `turbine` (D, E, F) |
| `difficulty` | `easy` / `medium` / `hard` (generator's judgement) |
| `derived_from`, `distractor_section_ids` | unanswerable questions only |
| `source`, `generator`, `corpus_version`, `review` | provenance and review status |

## Distribution
- **Types:** factual 31, numeric 26, procedural 15, list 14, definition 4, rationale 2, unanswerable 8.
- **Engine:** turbine 50, all 26, piston 24.
- **Difficulty:** easy 30, medium 56, hard 14.

## Set 2
A second, harder question set, all answerable (377 questions), is in `set2/`, written from the generation specs in `specs/`. It has four families:
- paraphrases of the 92 answerable questions;
- section-id queries;
- piston/turbine twin pairs;
- multi-section questions.

`specs/README.md` has the shared rules, the record format, the judge checks and the freeze procedure. `src/csrag/qgen.py` checks the files (`uv run python -m csrag.qgen check data/eval/set2/<family>.jsonl`). v1 above is unchanged.

## Which files an evaluation uses
`experiments.toml` at the repo root lists the question files of each experiment (`run1`: `questions.jsonl`; `run2`: `questions.jsonl` plus the four `set2/` files). The files are concatenated as they are; no merged file is written. Records without a `family` field (this set) are reported as family `set1`. The qrels of `questions.jsonl` alone are `qrels.jsonl` here; other experiments write theirs to `results/<experiment>/qrels.jsonl`.
