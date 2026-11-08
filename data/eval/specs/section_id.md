# Family: section_id

Read `README.md` in this folder first; its rules apply here too.

**What it measures:** questions that name a section id, the way experts search CS documents: "What is the objective of AMC E 510?", "Write me an extensive summary of CS-E 510". These questions test:
- the T1 id lookup;
- how BM25 handles an id: its tokeniser splits "CS-E 740(c)(4)" into `cs`, `e`, `740`, `c`, `4`;
- how dense models handle an id;
- for summaries, whether expansion collects a whole section.

**Size:** 50 questions. **Ids:** `id001`–`id050`. All have `mentions_section_id: true`.

**Family fields:**
- `id_form`: how the id is written;
- `id_intent`: what is asked.

## `id_form`: how the id is written

| form | what | example | T1's regex |
|---|---|---|---|
| `canonical` | the section id exactly as in the document, parenthesised parts included | "CS-E 510", "AMC E 20(f)", "AMC to CS-E 50(l)" | finds it |
| `paragraph` | a paragraph below a section id | "CS-E 740(c)(4)", "CS-E 510(g)(2)", "AMC E 515(3)" | finds the section ("CS-E 740(c)(4)" resolves to CS-E 740) |
| `loose` | the way people actually type ids: case, spacing or hyphen changed | "cs-e 510", "CSE 510", "CS E740", "AMC-E 510", "amc e 20(f)" | misses it, on purpose |

Loose forms change only case, spaces and hyphens. Keep the parentheses of a paragraph (write "cs-e 740(c)(4)", not "cse 740c4"): `qgen check` can resolve that, and then confirm that the id names the gold section.

## `id_intent`: what is asked

| intent | n | forms | gold | example |
|---|---|---|---|---|
| `objective` | 10 | 4 canonical, 3 paragraph, 3 loose | one quote: the sentence that states the objective, purpose or intent | "What is the objective of the safety analysis described in AMC E 510?" |
| `summary` | 10 | 5 canonical, 5 loose | 3–6 quotes from the one section, one per top-level paragraph, all `primary`; `requires_all_gold: true`; `question_type: "summary"` | "Write me an extensive summary of CS-E 510." |
| `lookup` | 20 | 12 paragraph, 4 canonical, 4 loose | one quote, inside the named section (and paragraph, if one is named) | "Under CS-E 740(c)(4), what must the alternate endurance test schedule be shown to provide?" |
| `redirect` | 10 | 6 canonical, 4 loose | one quote from an AMC of the named CS; the question does not name that AMC | "How can an applicant show compliance with CS-E 515?" → gold in AMC E 515 |

That gives 19 canonical, 15 paragraph and 16 loose.

- **objective:** the question names the section and asks for its objective, purpose, intent or scope. Paragraph forms name the paragraph that states it (e.g. "AMC E 20(f)(3)", the objective of the power availability procedures).
- **summary:** the answer needs the whole section. The quotes spread over the section and evidence recall measures how much of it was retrieved, so this is the test for parent expansion (T2) and the token budget. The reference `answer` is 4–8 bullets, each traceable to one paragraph.
- **lookup:** a specific requirement, number or condition from the named section or paragraph. With a paragraph id, the quote must lie inside that paragraph. `qgen check` cannot see paragraph boundaries, so check it by reading the section.
- **redirect:** the user names the CS, but the answer is in its AMC: the accepted means, the guidance, or what a term in the CS is taken to mean. Check B must label the CS NO or PARTIAL. If the CS itself answers, the question is a lookup, not a redirect.

Write the question as an expert would: short, direct, and naming the id. "Write me an extensive summary of …", "What is the objective of …", "What does … say about …" are all fine. Don't hide the id in a long sentence.

## Targets

- **At least 25 of the 50 gold sections have no v1 question.** Those 83 sections are: AMC E 10(b), CS-E 20, AMC E 20(f), AMC E 25, CS-E 30, AMC E 40, AMC E 40(b)(3) and (b)(4), AMC E 40(d), CS-E 50, AMC E 50, AMC E 50(e), AMC to CS-E 50(l), AMC E 60, AMC E 60(d), CS-E 70, AMC E 80, CS-E 130, AMC E 135, CS-E 140, AMC E 150(a), CS-E 170, AMC E 180, CS-E 210, AMC E 240, CS-E 300, AMC E 320, CS-E 340, AMC E 350, CS-E 380, AMC E 440(b)(4), CS-E 470, AMC E 500, CS-E 510, AMC E 515, CS-E 520, AMC E 520(a), AMC E 520(c)(2), AMC E 520(d), AMC E 525, CS-E 540, AMC E 560, CS-E 570, AMC E 600(e), CS-E 620, AMC E 640, CS-E 650, AMC E 660, CS-E 670, AMC E 680, CS-E 690, AMC E 700, CS-E 710, AMC E 720(a), CS-E 730, AMC E 740(c)(2)(i), AMC E 740(c)(3), AMC E 740(c)(4), AMC E 740(f)(1), AMC E 740(f)(5), AMC E 740(g)(1), AMC E 740(i)(2), CS-E 745, AMC E 750(b), CS-E 770, AMC E 780, CS-E 790, AMC E 790(a)(1), AMC E 790(a)(2), CS-E 790 Appendix A, AMC E 800, CS-E 810, AMC E 820(a)(2), CS-E 830, AMC E 840, CS-E 850, AMC E 870(a)(3), CS-E 890, AMC E 910, CS-E 920, AMC E 930, CS-E 1000, AMC E 1030, CS-E 1050.
- **Every subpart A–F has at least 3 questions.** Mix CS and AMC: at least 20 of each.
- **One question per section, unless the intents differ.** Do not ask a v1 question again with an id added; `qgen check` warns on near-duplicates.

**objective candidates:** these sections state an objective, purpose or intent; read them to confirm. AMC E 10(b), AMC E 20(f) (paragraph (3)), AMC E 25, AMC E 50 (paragraph (2)), AMC E 60(d) (paragraph (3)), AMC E 80, AMC E 130, AMC E 135, AMC E 170, AMC E 510 (paragraph (2)), AMC E 515, AMC E 740(c)(4), AMC E 740(f)(5), AMC E 780, CS-E 800 (paragraph (a)), CS-E 850 (paragraph (a)), AMC E 930, AMC E 1030.

**summary candidates:** these sections have 2–8 windows, so a whole one fits the 8k budget, and at least 3 top-level paragraphs. The number of windows is in brackets; sections with no v1 question are listed first.
- No v1 question: AMC E 25 (8), AMC E 40(b)(3) and (b)(4) (3), AMC E 40(d) (3), CS-E 50 (4), AMC E 50 (3), CS-E 130 (3), CS-E 140 (2), CS-E 300 (2), CS-E 340 (2), CS-E 510 (3), CS-E 520 (2), AMC E 560 (2), CS-E 570 (3), CS-E 650 (2), CS-E 790 (3), AMC E 840 (6), CS-E 890 (2), AMC E 930 (8).
- With a v1 question: CS-E 15 (3), AMC E 20 (2), CS-E 40 (2), CS-E 60 (2), CS-E 180 (2), CS-E 440 (7), CS-E 560 (2), AMC E 810 (5), CS-E 840 (3), AMC E 850 (5).

**redirect candidates:** any CS whose AMC holds the accepted means for the question. `data/groups.jsonl` lists each CS with its AMCs.

## Examples

Both pass `qgen check`.

```json
{"question_id": "id001", "question": "What is the objective of the safety analysis described in AMC E 510?",
 "question_type": "factual",
 "answer": "To ensure that the risk to the aircraft from all Engine Failure conditions is acceptably low, by managing the individual major and hazardous Engine risks to acceptable levels.",
 "answer_aliases": [], "answer_references": [],
 "gold": [{"section_id": "AMC E 510", "group_id": "CS-E 510", "quote": "The ultimate objective of a safety analysis is to ensure that the risk to the aircraft from all Engine Failure conditions is acceptably low.", "role": "primary"}],
 "requires_all_gold": false, "mentions_section_id": true, "difficulty": "easy", "generator": "<model id>",
 "family": "section_id", "asks_about_engine_type": "unspecified", "id_form": "canonical", "id_intent": "objective"}
```

```json
{"question_id": "id002", "question": "Write me an extensive summary of cs-e 510.", "question_type": "summary",
 "answer": "- (a) Analyse the Engine, including the control system, for the likely consequence of all Failures ... \n- (b) ...",
 "answer_aliases": [], "answer_references": [],
 "gold": [
  {"section_id": "CS-E 510", "group_id": "CS-E 510", "quote": "An analysis of the Engine, including the control system, must be carried out in order to assess the likely consequence of all Failures that can reasonably be expected to occur.", "role": "primary"},
  {"section_id": "CS-E 510", "group_id": "CS-E 510", "quote": "If significant doubt exists as to the effects of Failures and likely combination of Failures, any assumption may be required to be verified by test.", "role": "primary"},
  {"section_id": "CS-E 510", "group_id": "CS-E 510", "quote": "If reliance is placed on a safety system to prevent a Failure progressing to cause Hazardous Engine Effects", "role": "primary"},
  {"section_id": "CS-E 510", "group_id": "CS-E 510", "quote": "For compliance with CS-E, the following Failure definitions apply to the Engine:", "role": "primary"}],
 "requires_all_gold": true, "mentions_section_id": true, "difficulty": "hard", "generator": "<model id>",
 "family": "section_id", "asks_about_engine_type": "unspecified", "id_form": "loose", "id_intent": "summary"}
```

## Checks

Checks A and B from `README.md`.

- **summary:** check A gets the whole section. It passes if the judge's summary covers every paragraph the quotes come from. Check B passes if no other section is FULL. AMC E 510 is PARTIAL for a summary of CS-E 510.
- **redirect:** check B must not label the named CS FULL.

## Batches

| batch | ids | intents |
|---|---|---|
| 1 | id001–id020 | 10 objective, 10 summary |
| 2 | id021–id040 | 20 lookup |
| 3 | id041–id050 | 10 redirect |
