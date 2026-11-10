# Family: paraphrase

Read `README.md` in this folder first; its rules apply here too.

**What it measures:** how each method copes when the wording moves away from the regulation's, with the gold held fixed. Each paraphrase is paired with its v1 original, so the analysis can compare a method's rank of the same gold on the original and on the paraphrase. Only the wording changes between the two.

**Size:** 177 records, two for each answerable v1 question (`q001`–`q092`), one per level.

**Ids:** `pa<NNN><level>`, where `NNN` is the v1 number and the level is `h` (heavy) or `u` (user_voice). For example, `pa071h` is the heavy paraphrase of `q071`.

**Family fields:**
- `variant_of`: the original's id, e.g. `"q071"`;
- `paraphrase_level`: `"heavy"` or `"user_voice"`.

## Copied unchanged from the original

Copy `gold`, `answer`, `answer_aliases`, `answer_references`, `question_type` and `requires_all_gold` unchanged; `qgen check` compares them with the original. Set `asks_about_engine_type` from the paraphrase's own wording.

Section ids: the 6 v1 questions with `mentions_section_id: true` keep exactly the ids they name, and every other paraphrase names none.

## The two levels

| level | what it is | `lexical_overlap` |
|---|---|---|
| `heavy` | The same request in different words. Replace every content word of the gold section that has a plain alternative, and describe things instead of naming them with the regulation's term. | ≤ 0.50, and at least 0.20 below the original's |
| `user_voice` | How a practitioner who has not read the text would ask: plain register, first person or a short situation allowed, using the technical terms practitioners actually use. | ≤ 0.65, and at least 0.20 below the original's |

The v1 originals have overlap mean 0.84, median 0.88 and minimum 0.47 under this measure.

**How overlap is computed:** the share of the question's distinct content stems (Snowball, as in BM25) that occur anywhere in the gold section's text. Stopwords, one-letter tokens, "engine" and "CS"/"AMC"/"E" are not counted; numbers are. `qgen overlap "<draft>" "<gold section id>"` shows the stems that are still shared.

Some words have no plain alternative: "oil", "fuel", "bird", "hail", "propeller", "OEI", or the term a definition question asks about. They are fine, but they count toward the cap, so spend the rest of the question's words on alternatives.

## Keep the meaning

Every correct answer to the original must be a correct answer to the paraphrase, and the other way round.

**Keep:**
- the engine type, as words or as a cue from the list in `twin_topic.md` ("gas turbine", "reciprocating");
- the aircraft type (aeroplane or rotorcraft);
- the rating or test asked about;
- every condition and exception ("engines without a two-speed supercharger");
- what is asked for: the number, the list, the procedure or the reason, and every part of a two-part question;
- CS vs AMC. If the gold is an AMC, the paraphrase still asks for the accepted means, the guidance or the explanation. If the gold is a CS, it asks for the requirement.

**Don't:**
- narrow, widen or add a condition. "Turbofan" for "turbine engine" narrows it; "jet engine" narrows it too (turboshafts are turbine engines);
- add a fact the original does not state ("the 150-hour run");
- turn the question into a yes/no question, or split or merge questions;
- put the answer, or part of it, in the question;
- use made-up terms or non-standard abbreviations;
- write in another language.

## Examples

**q002** (overlap 1.00): "Which subparts of the engine certification specifications apply to turbine engines?" Gold: CS-E 10.
- `pa002h` (0.33): "Which parts of the rulebook are in force for gas-turbine powerplants?"
- `pa002u` (0.13): "My company builds gas turbines. Which chapters of these rules do we actually have to comply with?"

**q071** (0.88): "How many cold and hot starts must a turbine engine make during the endurance test, and how are 'cold' and 'hot' starts defined?" Gold: CS-E 750.
- `pa071h` (0.50): "On a gas turbine's long certification run, how many times must it be fired up from cold and how many from warm, and what separates the two?"
- `pa071u` (0.57): "We are planning the long endurance run for our new gas turbine. How many starts do we have to schedule, and what counts as a cold start versus a hot one?"

**q040** (0.79): "In the piston engine acceleration tests (engines without a two-speed supercharger), how many accelerations from idle to take-off power are required, and how fast is the power lever moved?" Gold: CS-E 390.
- Good, heavy (0.44): "For a reciprocating engine without a two-speed supercharger, how many times must it be run up from idle to full take-off output, and how quickly is the throttle pushed forward?"
- **Drifted, do not write:** "How quickly must a piston engine reach take-off power from idle, and how many times is this done?" CS-E 390 limits the time to move the lever (one second), not the time the engine takes to reach power. This question asks for something the text does not give, so it is unanswerable, and checks A and D both fail it. The two-speed supercharger exception has also been lost.

The record for `pa071h`, before `qgen fill` (fill adds `answerable`, `gold_kind`, `engine_type`, `subpart`, `lexical_overlap`, `corpus_version`, `source`, `review`, `verification`, `derived_from` and `distractor_section_ids`):

```json
{"question_id": "pa071h",
 "question": "On a gas turbine's long certification run, how many times must it be fired up from cold and how many from warm, and what separates the two?",
 "question_type": "numeric", "answer": "<copied from q071>", "answer_aliases": ["<copied>"], "answer_references": [],
 "gold": [{"section_id": "CS-E 750", "group_id": "CS-E 750", "quote": "<copied from q071>", "role": "primary"}],
 "requires_all_gold": false, "mentions_section_id": false, "difficulty": "hard", "generator": "<model id>",
 "family": "paraphrase", "asks_about_engine_type": "turbine", "variant_of": "q071", "paraphrase_level": "heavy"}
```

## Checks

Checks A, B and D from `README.md`. Check D is the one that catches meaning drift. A paraphrase that passes D but fails B has drifted toward another section: put back the condition that pinned the original to its gold.

## Batches

Write both levels of an original together. Eight batches:

| batch | originals | records |
|---|---|---|
| 1 | q001–q012 | 24 |
| 2 | q013–q024 | 24 |
| 3 | q025–q036 | 24 |
| 4 | q037–q048 | 24 |
| 5 | q049–q060 | 24 |
| 6 | q061–q072 | 24 |
| 7 | q073–q084 | 24 |
| 8 | q085–q092 | 16 |

If a paraphrase cannot meet its cap without changing the meaning, leave it out and say why in the batch's commit message. Do not loosen the meaning to meet the cap.
