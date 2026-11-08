# Family: twin_topic

Read `README.md` in this folder first; its rules apply here too.

**What it measures:** whether a method finds the section for the right engine type. CS-E covers many topics twice, once for piston engines (Subparts B and C) and once for turbine engines (Subparts D and E), sometimes in almost the same words. Each pair asks the same question twice, once per engine type, with different gold. This tests:
- the LLM engine-type filter F1, against the oracle F0 and no filter (H1);
- BM25 vs dense when the two texts barely differ.

**Size:** 30 pairs, 60 questions. **Ids:** `tw<NN><p|t>`, so `tw07p` and `tw07t` are the piston and turbine questions of pair 7. Number the pairs `tw01`–`tw30` in the order of the twin table below.

**Family fields:**
- `twin_key`: a key of the twin table;
- `twin_of`: the other question of the pair;
- `engine_cue`: `explicit` or `implied`;
- `asks_about_engine_type`: `piston` or `turbine`.

## Which section is the right source

This is where twin questions go wrong, so follow these rules exactly.

**Applicability (CS-E 10):** Subparts A, B and C apply to piston engines; Subparts A, D, E and F apply to turbine engines.

**Subpart A applies to both.** A question about either engine type can be answered by a Subpart A section. Never write a pair whose piston or turbine side is answered by Subpart A as well. These Subpart A sections are traps:
- **CS-E 100 Strength** (both types) next to **CS-E 520 Strength** (turbine only). A turbine strength question may be answered by both. Don't pair them.
- **CS-E 130 Fire Protection, CS-E 50 Engine Control System, CS-E 60 Provision for Instruments, CS-E 25 ICA, CS-E 140/150/160 Tests:** they have no piston/turbine twin. A question about them is answered by the same section for both types.
- **Paragraphs inside Subpart A that apply to one type only:** CS-E 15(c) (turbine terms) and (d) (piston terms); CS-E 40(b)(1) (piston ratings) and (b)(2)–(4) (turbine ratings); CS-E 60(d), (e) (turbine); CS-E 100(a), last sentence (turbine residual stresses); CS-E 110(d) (turbine); CS-E 130, the integral oil sump exception (piston); CS-E 180(b)(1)(i) (turbine) and (ii) (piston); AMC E 25 (6) (piston TBO); AMC E 30, Table 1 (turbine and piston rows); AMC E 40(d) (2) (piston) and (3) (turbine). Only the `common_section` keys below use these, and only as listed.

**Topics with no twin:** don't write pairs on these.
- **Piston only:** CS-E 230 induction icing (except the loose key `icing`), CS-E 240(a) spark ignition, CS-E 260 cooling, CS-E 290 hand turning, CS-E 300 test conditions, CS-E 330 tests general, CS-E 360 detonation, CS-E 450 ignition tests, CS-E 460 backfire.
- **Turbine only:** CS-E 500, 515, 520, 525, 540, 580, 590, 600, 640, 660, 680, 690, 700, 710, 720, 800, 810, 820, 840, 850, 860, 870, 880, 890, 900, 920, 930, and Subpart F (CS-E 1000–1050).

## The twin table

`src/csrag/qgen.py` (`TWINS`, or `qgen twins`) holds the authoritative section lists. A question's gold must come from its own side of its key: `qgen check` rejects anything else. The notes below say what each side covers. They were checked against `data/sections.jsonl`; read both sides in full before writing a pair.

**Classes:**
- **cross_subpart:** the two sides are different sections.
- **common_section:** one Subpart A section with a piston paragraph and a turbine paragraph. Both questions have the same gold section but quote different paragraphs. This checks that filters keep Subpart A sections.
- **loose:** related topics, not parallel texts. Optional, and needs extra care.

| key | pairs | piston side | turbine side | what each side says |
|---|---|---|---|---|
| `failure_analysis` | 2 | CS-E 210, AMC E 210 | CS-E 510, AMC E 510 | Both: failure analysis of the Engine including the control system, with installation assumptions stated. **Piston:** no single Fault, or double Fault where one may be present and undetected in pre-flight checks, may lead to unsafe conditions beyond the crew's control (210(a)). **Turbine:** assess all failures; Hazardous Engine Effects no more often than Extremely Remote, below 10^-7 per engine flight hour (510(a)(3)); a summary of failures with Major or Hazardous effects (510(a)(2)). |
| `fuel_system` | 2 | CS-E 250 | CS-E 560, AMC E 560 | 250(a) and 560(a)(1) (fuel specification declared and substantiated), and 250(c) and 560(b)(1) (filters and strainers sized for contaminants), are almost word for word the same. Only applicability tells them apart. **Turbine only:** fuel pump capacity margin (560(a)(3)); impending-blockage indication for the main filter (560(b)(2)). **Piston only:** no fuel draining into a stopped engine, "hydraulicing" (250(d)); protection of fuel control adjusting means (250(e)). |
| `oil_system` | 2 | CS-E 270 | CS-E 570, AMC E 570 | The oil filter paragraphs 270(c) and 570(b)(1) are almost the same. **Piston:** no oil draining into a stopped engine (270(a)); oil flow to the propeller control must not drop oil pressure below limits (270(b)); oil cooling (270(d)). **Turbine:** functioning in all flight attitudes (570(a)(1)); filler caps oil-tight and marked "oil" (570(a)(3)). |
| `relight_in_flight` | 1 | CS-E 240 | CS-E 910, AMC E 910 | Almost the same text: the constructor recommends a relight envelope and substantiates it (altitude, airspeed, windmilling speed, starter assistance, drill). The piston side is **240(b)(2), for self-ignition (compression-ignition, diesel) piston engines only**, and adds the effect of low ambient temperature. The piston question must be about a diesel engine: spark-ignition engines have no relight requirement. |
| `performance_correction` | 1 | CS-E 320, AMC E 320 | CS-E 620, AMC E 620 | Reference conditions for correcting performance results. **Piston:** the Standard Atmosphere, by an internationally recognised method (320(a)); AMC E 320 names ISO 1585. **Turbine:** 1013.25 hPa, 288 K and dry air (620(a)); no humidity correction to the power (620(b)). |
| `vibration` | 2 | CS-E 340, AMC E 340 | CS-E 650, AMC E 650 | Speed range of the vibration tests or surveys. **Piston:** crankshaft speeds from idle to the highest of 110% of Maximum Continuous speed, 105% of maximum take-off speed or the maximum over-speed, in 50 rpm steps, also with the most critical cylinder not firing (340(a)). **Turbine:** from minimum to at least 103% of the maximum speed for rating periods of 2 minutes or longer, 100% for shorter periods, and 100% of declared over-speeds (650(b)). |
| `calibration` | 2 | CS-E 350, AMC E 350 | CS-E 730, AMC E 730 | Calibration around the endurance test. **Piston:** sea-level power calibration curves at the beginning and end of the CS-E 440 test (350(b)); AMC E 350 gives the curves (against rotational speed and power setting). **Turbine:** thrust or power curves before and after the CS-E 740 test, or in its first and final stages, up to the highest rated powers except 30-Second and 2-Minute OEI (730); AMC E 730 gives the parameters (thrust, power, torque, speed, EPR, EGT). |
| `starting` | 2 | CS-E 370 | CS-E 750, AMC E 750(b) | Starts during the endurance test. **Piston:** at least 100 starts, half cold and half hot; time to start etc. recorded every 10 hours; at least 10 starts on each alternative means (370). **Turbine:** 25 cold starts (stopped ≥ 2 h) and 10 hot starts (within 15 min), 10 false starts, 100 in total (750). v1 already has q071 on the turbine cold/hot starts, so use another aspect, or ask something clearly different. |
| `low_temperature_starting` | 2 | CS-E 380, AMC E 380 | CS-E 770, AMC E 770 | **Piston:** at least 25 starts at oil inlet temperatures spread between +5 °C and the declared minimum; representative starting equipment; strip examination before and after (380). AMC E 380: declared minimum not higher than −20 °C. **Turbine:** starting and acceleration from the declared minimum temperatures, carcass and oil at the minimum, minimum and maximum starting torques, and a minimum oil temperature for acceleration (770). AMC E 770: uncooled intake air is acceptable. |
| `acceleration` | 2 | CS-E 390 | CS-E 745, AMC E 745 | **Piston:** five accelerations from idle to take-off at the end of the CS-E 440 test, power lever moved in ≤ 1 s, response without hesitation (390). **Turbine:** rated take-off reached when the lever is moved in ≤ 1 s (aeroplane: from minimum flight idle; rotorcraft: from minimum test-bed idle), and 15% to 95% of take-off in ≤ 5 s (745(a)). v1 has q040 (CS-E 390) and q070 (AMC E 745). |
| `over_speed` | 2 | CS-E 400 | CS-E 830, AMC E 830(c) | **Piston:** 20 runs of 30 s at the declared maximum over-speed or 5% above the maximum speed, whichever is greater (400(a)); a further 10 minutes in stages of at least 1 minute (400(b)). **Turbine:** 15 minutes at the Maximum Engine Over-speed, in runs of at least 2.5 minutes, at the maximum steady turbine-entry temperature and maximum oil temperature; the stripped condition must be satisfactory (830). |
| `endurance` | 2 | CS-E 440, AMC E 440(b)(4) | CS-E 740 and its 7 AMCs | **Piston:** a schedule of parts (Part 1 is a 30-hour run) and oil pressure settings (440). **Turbine:** 25 six-hour stages (740(c)), plus the alternate schedule for turbofans (740(c)(4)). Both are long (7 and 22 windows), so read them in full. |
| `contaminated_fuel` | 2 | CS-E 470, AMC E 470 | CS-E 670, AMC E 670 | **Piston:** no malfunction with solid contaminant, water and ice in the fuel; any reliance on anti-icing additive declared under CS-E 30 (470). AMC E 470: 1.0 g of solid contaminant per 1,000 litres, and more. **Turbine:** the maximum liquid and solid contamination likely in service; complete half the maximum flight duration after the first filter-blockage indication (670(b)(2)). AMC E 670: a contaminant table (e.g. 0.40 g/1000 litre magnetite). |
| `ratings` | 1 | CS-E 40 | CS-E 40 | common_section. Ratings besides Take-off and Maximum Continuous. **Piston:** Maximum Recommended Cruising and Maximum Best Economy Cruising (40(b)(1)). **Turbine:** OEI ratings; multi-engined aeroplanes: 2½-Minute and Continuous OEI (40(b)(2)). |
| `operating_limitations` | 1 | AMC E 40(d) | AMC E 40(d) | common_section. The list of operating limitations: (2) piston (e.g. cylinder and coolant temperatures), (3) turbine (e.g. RPM, indicated turbine gas temperature and time). |
| `propeller_pitch_change_cycles` | 1 | CS-E 180 | CS-E 180 | common_section. Pitch change cycles with a variable pitch propeller. **Piston:** 100 forward cycles across the range (180(b)(1)(ii)). **Turbine:** 50 forward cycles and 100 operations withdrawing the flight fine pitch lock (180(b)(1)(i)). |
| `installation_assumptions` | 1 | AMC E 30 | AMC E 30 | common_section. Table 1 has a TURBINE ENGINES block and a PISTON ENGINES block. Quote one row from each. |
| `icing` | 1 | CS-E 230 | CS-E 780, AMC E 780 | loose. **Piston:** induction system design against ice, and provision for an induction thermometer or ice indicator (230). **Turbine:** tests in icing conditions (780). Only a broad question pairs these ("what does CS-E require about ice for …?"); don't ask for details. |
| `water_ingestion` | 1 | CS-E 430 | CS-E 790 and its AMCs, Appendix A | loose. **Piston:** water spray test, three running periods with 24-hour intervals (430). **Turbine:** rain and hail ingestion (790). |

That is 24 cross_subpart pairs, 4 common_section pairs and 2 loose pairs: 30 in all. `qgen check` warns when a key goes over its listed quota.

## Writing a pair

1. Read both sides of the key in full and pick an aspect that both sides cover. Write the piston and turbine questions from **one template**: change only the engine-type wording.
   - Where the texts differ, the answers differ.
   - Where they are the same (fuel specification, the filters, relight), the answers are the same too, and only applicability separates the gold. These are the cleanest filter tests; note "identical text" in `verification.notes`.
2. **Both questions use the same `engine_cue`.** Use `explicit` for half the pairs and `implied` for the other half. Where a key has two pairs, give it one of each.
   - `explicit`: the question says "piston" or "turbine".
   - `implied`: the question uses neither word, nor "reciprocating", and names the engine type through a cue word instead. `qgen check` enforces this.

   | piston cues | turbine cues |
   |---|---|
   | crankshaft, cylinder(s), spark plug(s), magneto, carburettor, mixture setting, supercharger, manifold or boost pressure, diesel or compression-ignition (only for 240(b)) | turbofan, turbojet, turboprop, turboshaft, gas generator, fan, compressor surge or stall, exhaust gas temperature, OEI rating, propelling nozzle, thrust. Not "gas turbine": it contains the word "turbine", so it counts as explicit |

   - A cue must not narrow the question beyond its gold. "Turbofan" is wrong for a requirement on all turbine engines unless the gold says turbofan (740(c)(4) does).
   - "Rotorcraft" and "propeller" are not cues: both types have them.
3. **No section ids and no subpart names.** Write "piston engine", never "Subpart C".
4. `answer`: each side's own answer.

## Example (passes `qgen check`)

```json
{"question_id": "tw01p", "question": "In the piston engine over-speed tests, how many runs are made and how long does each one last?",
 "question_type": "numeric",
 "answer": "20 runs, each of 30 seconds, at the declared Maximum Engine Over-speed or at 5% above the declared maximum rotational speed, whichever is greater (engines other than two-speed supercharged ones).",
 "answer_aliases": ["20 runs of 30 seconds"], "answer_references": [],
 "gold": [{"section_id": "CS-E 400", "group_id": "CS-E 400", "quote": "All Engines, except two-speed supercharged Engines. 20 runs, each of 30 seconds duration", "role": "primary"}],
 "requires_all_gold": false, "mentions_section_id": false, "difficulty": "medium", "generator": "<model id>",
 "family": "twin_topic", "asks_about_engine_type": "piston", "twin_key": "over_speed", "twin_of": "tw01t", "engine_cue": "explicit"}
{"question_id": "tw01t", "question": "In the turbine engine over-speed test, how many runs are made and how long does each one last?",
 "question_type": "numeric",
 "answer": "A total of 15 minutes at the Maximum Engine Over-speed to be approved, which may be split into separate runs of at least 2.5 minutes each.",
 "answer_aliases": ["15 minutes in runs of at least 2.5 minutes"], "answer_references": [],
 "gold": [{"section_id": "CS-E 830", "group_id": "CS-E 830", "quote": "A total of 15 minutes run at the Maximum Engine Over-speed to be approved. This may be done in separate runs, each being of at least 2.5 minutes duration.", "role": "primary"}],
 "requires_all_gold": false, "mentions_section_id": false, "difficulty": "medium", "generator": "<model id>",
 "family": "twin_topic", "asks_about_engine_type": "turbine", "twin_key": "over_speed", "twin_of": "tw01p", "engine_cue": "explicit"}
```

(The example's numbering is only illustrative. In the real file, pairs are numbered in table order, so `over_speed` gets tw19–tw20.)

## Checks

Checks A and B from `README.md`. A twin's pool always contains both sides of its key. Check B passes only if:
- the twin's section is labelled NO, because it is the wrong engine type;
- no Subpart A section is labelled FULL (except the gold, for a common_section pair).

If a Subpart A section is FULL, the pair is on a shared topic: pick another aspect.

## Batches

| batch | pairs | keys |
|---|---|---|
| 1 | tw01–tw10 | `failure_analysis` to `vibration` |
| 2 | tw11–tw20 | `calibration` to `over_speed` |
| 3 | tw21–tw30 | `endurance` to `water_ingestion` |

Run `qgen check` on the whole file, so that it sees both questions of every pair.
