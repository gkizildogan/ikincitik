# Family: multi_section

Read `README.md` in this folder first; its rules apply here too.

**What it measures:** questions that need evidence from two or more sections. In v1 every question had one gold section, so this part of RQ6 was never tested: the multi-section metrics (`supporting_recall@k`, `all_gold@k`) had n = 0. This family tests:
- fixed parent expansion (T2) and cross-reference following (T3, T2+T3);
- the agent (T4), with `expand`, `follow_refs` and `search`;
- whether first-stage retrieval and rerankers bring in both sections.

**Size:** 90 questions, all `requires_all_gold: true`. **Ids:** `ms001`–`ms090`.

**Family fields:**
- `hop_type`: `cs_amc`, `reference` or `comparison`;
- `twin_key`, comparison only: the key of the twin table in `twin_topic.md`.

| hop_type | n | ids | what links the sections | roles |
|---|---|---|---|---|
| `cs_amc` | 30 | ms001–ms030 | a CS and its own AMC (same group) | CS `primary`, AMC `supporting` |
| `reference` | 40 | ms031–ms070 | section 1 cites section 2 (`data/cross_refs.jsonl`), in different groups | citing section `primary`, cited section `supporting` |
| `comparison` | 20 | ms071–ms090 | the piston and turbine sides of one twin key | both `primary` |

Each gold section gets at least one quote, the sentence it contributes to the answer. The question names no section id, in any spelling, and no subpart.

## Every gold section must be needed

The family exists for this rule. A question passes only if:
- all its gold sections together answer it fully (check A);
- **no gold section answers it alone** (check C);
- no other section answers it alone (check B).

A two-part question ("what is required, and how is it shown?") passes only when each part's answer is in a different section. If the CS already says how it is shown, it is not a multi_section question.

## `cs_amc` (30)

The CS states the requirement, and its AMC gives what the CS leaves open: the accepted means, the test levels, the method, or the meaning of a term. The question asks for both in one natural request. For example: "What must be shown about X, and what levels are accepted for that demonstration?"

- **Targets:** 64 groups have a CS and at least one AMC; `data/groups.jsonl` lists each group's sections. Use at most one cs_amc question per group, and spread the 30 over Subparts A to F.
- With several AMCs in a group (CS-E 520, CS-E 740, CS-E 790), the gold is the CS and one AMC, unless the question really needs two.

## `reference` (40)

The citing section points to another section for something the answer needs: "the endurance test of CS-E 440", "as defined in CS-E 510(g)", "the conditions of CS-E 790". The question is about the citing section's topic, and needs the cited content to be complete.

**Candidates:** `qgen refs` lists all 263 cross-group references, one per (citing, cited) pair, each with the citing sentence. `qgen refs --section "CS-E 740"` lists those from or to one section. `(See AMC E …)` pointers are already removed.

**Rules:**
- Use each (citing, cited) pair at most once.
- Use a section at most twice as the citing one and at most three times as the cited one. Several sections are cited very often: CS-E 740 (23 times), CS-E 20 (20), CS-E 510 (17), CS-E 30 (11), CS-E 440 (11).
- Use the general sections CS-E 15 (terminology), CS-E 20 and CS-E 30 as the cited section in at most 6 questions in total.
- Skip rows of the AMC E 30 and AMC E 80 tables: a table row only names the section.
- `qgen check` confirms that the primary cites the supporting section in `data/cross_refs.jsonl`, and that the two are in different groups. Within a group, use `cs_amc`.

**Example** (passes `qgen check`; CS-E 730 cites CS-E 740(h)):

```json
{"question_id": "ms031",
 "question": "Turbine thrust or power calibration curves are used to check the engine around the endurance run. When must they be taken, and what must the engine still be able to deliver once that run is over?",
 "question_type": "procedural",
 "answer": "Either by specific tests immediately before and after the endurance test, or from measurements in its first and final stages, up to the highest rated powers except the 30-Second and 2-Minute OEI ratings. After the test the Engine must still provide the rated thrust or power levels without exceeding operating limitations.",
 "answer_aliases": [], "answer_references": [],
 "gold": [
  {"section_id": "CS-E 730", "group_id": "CS-E 730", "quote": "thrust or power calibration curves of the test Engine must be established either by specific tests accomplished immediately before and after the endurance test or by measurements obtained during the first and final stages of the endurance", "role": "primary"},
  {"section_id": "CS-E 740", "group_id": "CS-E 740", "quote": "After completion of the endurance test, the Engine must be able to provide the rated thrust or power levels without exceeding operating limitations.", "role": "supporting"}],
 "requires_all_gold": true, "mentions_section_id": false, "difficulty": "hard", "generator": "<model id>",
 "family": "multi_section", "asks_about_engine_type": "turbine", "hop_type": "reference"}
```

CS-E 730 alone says only that "performance targets are met (as per CS-E 740(h))", not what the targets are. CS-E 740 alone does not say when the calibration curves are taken. Check C must confirm both.

## `comparison` (20)

The question asks how a requirement differs between piston and turbine engines, and the answer needs both sides. It may say "piston" and "turbine", since that is the point of the question. Set `asks_about_engine_type: "both"`.

- **Gold:** one section from each side of a twin key in `twin_topic.md`, both `primary`, with `twin_key` set. `qgen check` confirms that one section comes from each side.
- Use every cross_subpart key at least once, and none more than twice. The loose keys `icing` and `water_ingestion` may also be used once each.
- Prefer an aspect other than the one used for the twin_topic pair of the same key.
- Don't use a common_section key here: those have one gold section.

Example: "How do the over-speed test runs required for type certification differ between piston and turbine engines?" Gold: CS-E 400(a) (20 runs of 30 s) and CS-E 830(b)(1) (15 minutes in runs of at least 2.5 minutes).

## Checks

Checks A, B and C from `README.md`.

- **A:** the judge gets all the gold sections.
- **C:** one run per gold section, each alone; every run must return `fully_answered: false`.
- **B:** passes if no section outside the gold is FULL and every gold section is FULL or PARTIAL. A gold section labelled FULL in B also fails C.

## Batches

| batch | ids | hop_type |
|---|---|---|
| 1 | ms001–ms015 | cs_amc |
| 2 | ms016–ms030 | cs_amc |
| 3 | ms031–ms050 | reference |
| 4 | ms051–ms070 | reference |
| 5 | ms071–ms090 | comparison |
