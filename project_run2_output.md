# CS-E retrieval experiments: run 2 results (477 questions, v1 parameters)

**Scope:** experiment `run2` in `experiments.toml`: the same 110 runs as run 1 (55 per index), with parameter set `v1` unchanged, on 477 questions. 469 are answerable and scored; the 8 unanswerable ones are excluded from scoring. The questions are question set 1 (100 questions, the run 1 set) plus question set 2 (377 answerable questions in four families). Numbers come from `results/run2/units/metrics.*`, `results/run2/win500/metrics.*`, `results/run2/granularity.csv` and the two parquet files, which notebook 09 wrote. This report covers retrieval only.

**Run 1 is contained in run 2.** The `set1` family is the run 1 question set. For all 110 runs, its Hit@1, MRR@10 and nDCG@10 in `results/run2/` are identical to the archived run 1 values in `run1resultarchive/`. So every `set1` column below is the run 1 result, and nothing in `project_v1_output.md` is contradicted. Run 2 adds what run 1 could not measure.

**How this document relates to the plan:** `plan.md` says no run is selected as "the winner". The RQ sections (4 to 10) keep to that: they report each pre-declared test and name the leading option as a description only. Section 12, "Winner and other combinations", is a separate engineering decision made after seeing the results on the same questions. It should not be presented as a pre-declared finding.

## 0. Terms

### 0.1 Methods

#### Single-signal baselines (RQ1)
| id | method | Qdrant mechanism |
|---|---|---|
| L1 | BM25 | `query(bm25 sparse)` |
| S1 | bge-m3 learned sparse | `query(bge_sparse)` |
| D1 | bge-m3 dense | `query(dense)` |
| C1 | ColBERT MaxSim over all units | exact MaxSim over every unit's full ColBERT vectors, client-side (Qdrant's multivector cap, plan section 3.5) |

#### Hybrid and multi-stage (RQ2)
| id | method |
|---|---|
| H1 | BM25 + dense: each top-50 → gate A → **RRF** |
| H2 | BM25 + bge-sparse + dense: each top-50 → gate A → RRF (three-way) |
| M1 | dense top-50 → gate A → **ColBERT rerank** of the kept hits (exact MaxSim, client-side) → gate B |

#### Metadata and structure (RQ5), on top of H1
| id | method |
|---|---|
| F0 | **oracle filter**: the same filter as F1, but built from the question's `engine_type` label instead of the LLM. `piston` / `turbine` → `engine_type ∈ {label, all}`; `all` → no filter. It is the upper bound for F1, and the only method that reads a label. |
| F1 | **LLM-extracted filter**: Qwen returns JSON (schema `{engine_type: "piston" \| "turbine" \| null}`), and the filter becomes `engine_type ∈ {extracted, all}`; `null` means no filter. The filter is applied inside each signal's Qdrant search, before gate A. |
| T1 | **id lookup**: a regex finds "CS-E 740", "AMC E 20(f)" in the question text; that section's units or windows go first, then H1's results |

#### Query-side LLM methods (RQ4; cached in `data/cache/llm.jsonl`)
| id | method |
|---|---|
| Q1 | **HyDE**: Qwen writes one hypothetical CS/AMC-style passage that answers the question. The dense query vector is the normalised mean of the question's and the passage's embeddings. Dense search → gate A; compared with D1. |
| Q2 | **keyword expansion for BM25**: Qwen lists regulation-style terms for the question. BM25 runs on the question plus these terms → gate A; compared with L1. |

#### Rerankers (RQ3), on H1's fused list
| id | method |
|---|---|
| X1 | **bge-reranker-v2-m3**: pointwise cross-encoder, `encoder-gpu` (phase A) → gate B |
| X2 | **jina-reranker-v3**: listwise (the query and all candidates in one context), `encoder-gpu` (phase A) → gate B |

#### Expansion: fixed rules and agent (RQ6), on H1
| id | method |
|---|---|
| T0 | **no expansion**: H1's units or windows in rank order, overlapping windows merged, cut to the token budget (the reference point for every expansion method) |
| T2 | **fixed parent expansion** (small-to-big): H1's hits in rank order → their sections (or groups), deduplicated, until the token budget is full |
| T3 | **fixed cross-reference following**: H1's hits in rank order, each followed by the sections it references, until the token budget is full. It always includes the AMCs of a retrieved CS. |
| T2+T3 | both fixed rules together, under the same token budget |
| T4 | **agentic small-to-big**. It starts from H1's top 5 hits (gate on). Qwen gets the tools `expand(unit_id, to="section"\|"group")`, `follow_refs(unit_id)`, `search(query)` and `done(evidence_ids)`, at most 5 steps, within the token budget. It decides per question whether the hits are enough (it calls `done` straight away) or whether to expand, follow a reference or search again. |

#### Everything in a single list
| id | method |
|---|---|
| L1 | BM25 |
| S1 | bge-m3 learned sparse |
| D1 | bge-m3 dense |
| C1 | ColBERT MaxSim over all units |
| H1 | BM25 + dense, RRF |
| H2 | BM25 + bge-sparse + dense, RRF |
| M1 | dense, then ColBERT rerank |
| F0 | oracle engine-type filter on H1 (reads the label; upper bound only) |
| F1 | LLM engine-type filter on H1 |
| T1 | section-id lookup in front of H1 |
| Q1 | HyDE, dense |
| Q2 | LLM keyword expansion, BM25 |
| X1 | H1, then bge-reranker-v2-m3 |
| X2 | H1, then jina-reranker-v3 |
| T0 | H1, no expansion, cut to a token budget |
| T2 | H1, then parent sections |
| T3 | H1, then cross-referenced sections |
| T2+T3 | both fixed rules |
| T4 | H1, then the Qwen agent |

### 0.2 The 477 questions

All questions were generated by Claude (`claude-opus-5-5`). A question's family is its `family` field; set 1 has none and is reported as `set1`.

| Family | Scored | What the question looks like | Gold | What it was written to test |
|---|---|---|---|---|
| `set1` | 92 (+ 8 unanswerable, not scored) | One question per group, in the regulation's own wording: on average 84% of a question's content stems occur in its gold section. | one section | the run 1 baseline |
| `paraphrase` | 177 | Rewrites of 89 of the 92 set 1 questions, with the same gold. 89 are **heavy** (every content word replaced where a plain alternative exists; mean overlap 0.22) and 88 are **user voice** (how a practitioner who has not read the text would ask; mean overlap 0.45). | one section, the original's | wording sensitivity: BM25 against dense and ColBERT, HyDE, keyword expansion, rerankers |
| `section_id` | 50 | The question names a section id. The id is written canonically (19), with a paragraph (15, "CS-E 740(c)(4)") or loosely (16, "cse 730"). The intent is a lookup (20), an objective (10), a **redirect** (10: the question names the CS, the answer is in its AMC) or a **summary** (10: 3 to 6 quotes spread over one section, all needed). | one section | id lookup T1, and how BM25 and dense treat an id |
| `twin_topic` | 60 | 30 pairs. Each pair asks the same thing once for piston and once for turbine engines. In 52 questions the two sides have different gold sections; in 8 they share one Subpart A section and differ by paragraph. The engine type is named outright in 30 questions and only implied in 30 ("an engine with cylinders and a crankshaft"). | one section | the engine-type filter F1 against F0 and H1 |
| `multi_section` | 90 | The answer needs two sections, and neither is enough alone: a CS and its AMC (`cs_amc`, 30), a section and one it cites in another group (`reference`, 40), or the piston and turbine sides of one topic (`comparison`, 20). | two sections | expansion T2, T3 and the agent T4 |

Examples:

- `set1` → `paraphrase`: "Which subparts of the engine certification specifications apply to turbine engines?" becomes "Which parts of the rulebook are in force for gas-turbine powerplants?" (heavy) and "My company builds gas turbines. Which chapters of these rules do we actually have to comply with?" (user voice). Gold stays CS-E 10.
- `section_id`, redirect: "What are the acceptable means of establishing the susceptibility required by CS-E 1050?" The gold is AMC E 1050.
- `section_id`, loose summary: "give me a detailed summary of cs-e 130".
- `twin_topic`, implied cue: "Which combinations of failures does the failure analysis of an engine with cylinders and a crankshaft have to take into account?" (gold CS-E 210) and the same question for "a turbojet, turbofan, turboprop or turboshaft engine" (gold CS-E 510).
- `multi_section`, reference: "Once a spark-ignition piston engine has finished its endurance test, what must be done to check its tendency to backfire, and what inspection and re-measurement must follow that test?" Gold: CS-E 460 and CS-E 440.

### 0.3 Metrics

| Metric | Meaning |
|---|---|
| Hit@1, Hit@3 | A gold section is at rank 1 (in the top 3). Hits are deduplicated to sections first. For a `multi_section` question either gold section counts. |
| MRR@10 | 1 / rank of the first gold section, 0 if it is not in the top 10. |
| nDCG@10 | Graded ranking score at section level: grade 2 for a gold section, grade 1 for another section of the same group (usually the CS/AMC sibling). It is the plan's sort key. |
| Evidence@k | Share of a question's gold quotes whose text is covered by the top-k retrieved units or windows. It is independent of the chunking, and the strictest measure of "is the answer text there". With several quotes (multi-section, summary) it can be fractional. |
| Evidence (context) | The same, over everything the run hands over: the whole returned list for ranked runs (at most 20 hits), the content inside the token budget for T0 to T3, and the agent's working context for T4. |
| All-gold@k | `multi_section` only: every gold section is in the top k. |
| Supporting recall@k | `multi_section` only: the supporting section (the AMC, or the cited section) is in the top k. |
| Complete evidence in context | `multi_section` only: share of questions with every gold quote in the context. |
| Gate loss | Share of questions whose gold section is in the gate-off list of the same configuration but not in the gate-on list. |
| Context tokens | bge-m3 tokens handed over per question, after merging overlapping windows. |
| Latency | Seconds per question on the RTX 3090. |

For T4 the ranked list is the agent's `done(evidence_ids)` selection first, then everything else it retrieved. Its Hit@1 therefore says "the first passage the LLM selected is from a gold section".

## 1. Summary

| RQ | Question | Short answer in run 2 | Leading option | Pre-declared tests (Holm-corrected) |
|---|---|---|---|---|
| RQ1 | Best single signal? | **Dense and ColBERT beat BM25.** The whole gain is on paraphrased questions; on set 1 wording nothing differs. Learned sparse is no better than BM25. | ColBERT (C1), with dense (D1) close behind at a fraction of the latency | D1 and C1 vs L1 significant on every metric, on both indexes. S1 vs L1 not better. |
| RQ2 | Does hybrid beat its parts? | **It beats BM25, not dense.** On heavy paraphrases the BM25 half pulls the fused list below dense alone. | H1 as a candidate generator, not as a final ranking | H1 vs L1 significant everywhere. H1 vs D1 not significant with the gate off. H2 and M1 add nothing. |
| RQ3 | Do rerankers help, and which one? | **Yes, and the two are no longer tied.** jina-reranker-v3 (X2) beats H1 and beats bge-reranker-v2-m3 (X1) on every metric. X1 helps on windows only. | X2 | X2 vs H1 and X1 vs X2 significant in all four index and gate settings. X1 vs H1 significant on windows only. |
| RQ4 | Do LLM query rewrites help? | **HyDE now helps a little; keyword expansion still hurts overall.** Both gains are confined to paraphrases. | Q1 over plain dense, at 13 s per query and still far below X2 | Q1 vs D1 significant on MRR, nDCG and evidence. Q2 vs L1 significantly worse on units. |
| RQ5 | Do metadata and structure help? | **The engine-type filter does, for twin questions.** The id lookup does not: it fixes summaries and breaks redirects. | F1 | F1 vs H1 significant on Hit@1 in three of four settings. T1 vs H1 only a negligible nDCG gain. |
| RQ6 | Does expansion help? Is the agent better? | **Fixed expansion does not add evidence, even on multi-section questions.** T3 does find the cited section, but pushes other evidence out. **The agent beats the fixed rules on every metric.** | Agent (T4) | T4 vs T2+T3 significant on Hit@1, MRR and nDCG in all six settings. T3 and T2+T3 significantly lower MRR everywhere. |
| RQ7 | Windows or units? | **Depends on the signal.** Windows clearly help the bge reranker and slightly help ColBERT. They cost BM25, dense and H1 a little evidence. | Windows when a reranker follows; they also halve the context | X1 significant on every ranking metric. C1 on MRR. L1 and D1 significantly lower Evidence@10. |

**Winner (section 12).** Two runs lead, and each wins one criterion with a significant margin over the other:

- **Evidence coverage, and the best deterministic layout: X2 on windows without the gate.** BM25 + bge-m3 dense (top 50 each) → RRF → jina-reranker-v3 → keep the top 10. Evidence@10 0.981 (0.984 over all 20 returned), Hit@1 0.853, Hit@3 0.970, about 3,800 tokens for the top 10, 8.2 s per question, no LLM. This is my default pick.
- **Early precision: the agent T4 on windows at the 8k budget.** Hit@1 0.925, MRR@10 0.948, the best in every family. Evidence in context 0.951 in about 2,400 tokens, 14.7 s and 1.5 LLM calls per question.

Compared with X2, T4 has +0.072 Hit@1 and −0.033 evidence in context; both differences are significant. They are not competing designs: T4 as run starts from H1's top 5, not from X2's list. That combination was not tested.

**What changed since run 1.** On set 1 wording the leading methods sat near a ceiling (Hit@1 0.90 to 0.98) and could not be separated. With 469 scored questions pooled Hit@1 ranges from 0.52 to 0.93, and several run 1 readings no longer hold:

| Run 1 reading | Run 2 |
|---|---|
| No signal clearly beats BM25 | Dense and ColBERT beat BM25. BM25's Hit@1 falls from 0.84–0.86 on set 1 to 0.39–0.41 on paraphrases. |
| X1 and X2 are tied | X2 is ahead by 0.08 (windows) to 0.14 (units) Hit@1. On heavy paraphrases X1 on units (0.315) is below dense alone (0.494). |
| HyDE adds nothing | HyDE adds +0.03 to +0.04 MRR and evidence over dense. |
| The gate costs nothing after fusion or reranking | The gate loses the gold section on 2% (H1) to 4.5% (X1) of questions, and caps evidence coverage at 0.89–0.95. |
| The engine-type filter barely matters | It lifts twin-question Hit@1 from 0.45–0.52 to 0.68–0.70. |
| The reranker's top score flags unanswerable questions | It no longer does: 77% of paraphrases score below the highest unanswerable question under X1. |
| Default combination: X1 on windows, gate on | Still 0.967 Hit@1 on set 1, but 0.768 pooled and 0.667 on paraphrases. |

**Biggest caveats:** all 477 questions are LLM-generated; the four new families are designed stress tests, so the pooled numbers depend on the family mix (paraphrases are 38% of the scored questions); and a paraphrase shares its gold with its original, so pooled confidence intervals and tests are narrower than they should be (section 14).

## 2. How to read "best"

No single number decides this. I read the results in this order:

1. **Is the evidence retrieved at all?** Evidence recall: the gold quote's text is covered by the retrieved chunks. It compares fairly across both indexes.
2. **Is the right section near the top?** Hit@1 and MRR@10 at section level. They count only true gold sections.
3. **What does it cost?** Context tokens, latency, and whether an LLM is needed.
4. **Is a difference real?** A difference counts as a finding only if its Holm-corrected p-value is below 0.05 on a pre-declared comparison. With 469 questions the 95% interval on a pooled Hit@1 is about ±0.02 to ±0.04, so pooled differences of 0.05 and more are now resolvable. Inside one family (50 to 177 questions) they often are not.

**Pooled or per family.** The pooled column (`all`) weights each family by its size. It answers "how does this method do on this particular mix", and the mix was designed, not sampled from users. For each RQ I report the pooled pre-declared tests first, because those are the plan's tests, and then say which families carry the result. The per-family tests use the same paired randomisation test with Holm correction inside each family, RQ, metric and index, but they were **not pre-declared**. Treat them as explanation, not as findings.

**nDCG@10 is still not the number to sort by.** Section nDCG gives partial credit to sibling sections in the gold's group, so methods that add or keep siblings gain nDCG without finding the gold any better. Example from run 2: T3 on windows (8k, gate on) has a higher nDCG than T0 (0.762 against 0.743) while its MRR is 0.034 lower and its evidence in context is lower too. The tables show nDCG with its interval because the plan asks for it.

**Table conventions.** Hit@1, MRR and nDCG are at section level. "Units" are the 252 structure-bounded base units, "windows" the 436 section-bounded 500-token windows. Δ is the first method minus the second. **Bold** means Holm-corrected p < 0.05. Latency is per question on the RTX 3090. Unless a table says otherwise, the gate is off, as in the run 1 report.

**Method codes:** L1 BM25 · S1 bge-m3 learned sparse · D1 bge-m3 dense · C1 ColBERT · H1 BM25 + dense RRF · H2 BM25 + sparse + dense RRF · M1 dense then ColBERT rerank · X1 bge-reranker-v2-m3 · X2 jina-reranker-v3 · Q1 HyDE · Q2 keyword expansion · F0 oracle engine-type filter · F1 LLM engine-type filter · T1 section-id lookup · T0 no expansion · T2 parent section · T3 cross-references · T4 agent.

## 3. Results by question family

These two tables are the quickest way to see run 2. Each row is one run; the columns are the families. T4 exists with the gate on only.

**Hit@1 (section level)**

| Method | Index | Gate | all (469) | set1 (92) | paraphrase (177) | section_id (50) | twin_topic (60) | multi_section (90) |
|---|---|---|---|---|---|---|---|---|
| L1 | units | off | 0.586 | 0.837 | 0.412 | 0.740 | 0.300 | 0.778 |
| S1 | units | off | 0.539 | 0.761 | 0.339 | 0.860 | 0.367 | 0.644 |
| D1 | units | off | 0.659 | 0.826 | 0.576 | 0.800 | 0.483 | 0.689 |
| C1 | units | off | 0.672 | 0.815 | 0.621 | 0.780 | 0.367 | 0.767 |
| H1 | units | off | 0.680 | 0.891 | 0.497 | 0.880 | 0.517 | 0.822 |
| H2 | units | off | 0.652 | 0.848 | 0.486 | 0.860 | 0.450 | 0.800 |
| M1 | units | off | 0.672 | 0.815 | 0.621 | 0.780 | 0.367 | 0.767 |
| Q1 | units | off | 0.699 | 0.826 | 0.650 | 0.760 | 0.483 | 0.778 |
| Q2 | units | off | 0.518 | 0.685 | 0.458 | 0.520 | 0.283 | 0.622 |
| F0 | units | off | 0.710 | 0.891 | 0.508 | 0.860 | 0.700 | 0.844 |
| F1 | units | off | 0.704 | 0.891 | 0.497 | 0.840 | 0.700 | 0.844 |
| T1 | units | off | 0.684 | 0.891 | 0.514 | 0.860 | 0.517 | 0.822 |
| X1 | units | off | 0.695 | 0.902 | 0.537 | 0.940 | 0.483 | 0.800 |
| X2 | units | off | 0.838 | 0.946 | 0.814 | 0.960 | 0.550 | 0.900 |
| T4 8k | units | on | 0.928 | 0.978 | 0.893 | 0.980 | 0.817 | 0.989 |
| L1 | windows | off | 0.595 | 0.859 | 0.390 | 0.680 | 0.367 | 0.833 |
| S1 | windows | off | 0.561 | 0.804 | 0.345 | 0.820 | 0.400 | 0.700 |
| D1 | windows | off | 0.667 | 0.837 | 0.610 | 0.740 | 0.433 | 0.722 |
| C1 | windows | off | 0.697 | 0.891 | 0.621 | 0.840 | 0.367 | 0.789 |
| H1 | windows | off | 0.646 | 0.848 | 0.486 | 0.800 | 0.450 | 0.800 |
| H2 | windows | off | 0.648 | 0.859 | 0.480 | 0.840 | 0.450 | 0.789 |
| M1 | windows | off | 0.697 | 0.891 | 0.621 | 0.840 | 0.367 | 0.789 |
| Q1 | windows | off | 0.708 | 0.826 | 0.672 | 0.720 | 0.517 | 0.778 |
| Q2 | windows | off | 0.557 | 0.728 | 0.475 | 0.540 | 0.350 | 0.689 |
| F0 | windows | off | 0.689 | 0.859 | 0.508 | 0.800 | 0.683 | 0.811 |
| F1 | windows | off | 0.687 | 0.870 | 0.508 | 0.760 | 0.683 | 0.811 |
| T1 | windows | off | 0.655 | 0.848 | 0.497 | 0.840 | 0.450 | 0.800 |
| X1 | windows | off | 0.774 | 0.967 | 0.684 | 0.900 | 0.500 | 0.867 |
| X2 | windows | off | 0.853 | 0.946 | 0.859 | 0.940 | 0.517 | 0.922 |
| T4 8k | windows | on | 0.925 | 0.957 | 0.932 | 0.980 | 0.783 | 0.944 |

**Evidence@10**

| Method | Index | Gate | all (469) | set1 (92) | paraphrase (177) | section_id (50) | twin_topic (60) | multi_section (90) |
|---|---|---|---|---|---|---|---|---|
| L1 | units | off | 0.810 | 0.978 | 0.621 | 0.980 | 0.917 | 0.842 |
| S1 | units | off | 0.792 | 0.989 | 0.633 | 0.940 | 0.850 | 0.782 |
| D1 | units | off | 0.912 | 0.967 | 0.876 | 0.960 | 0.933 | 0.887 |
| C1 | units | off | 0.908 | 0.989 | 0.870 | 0.940 | 0.900 | 0.889 |
| H1 | units | off | 0.901 | 0.989 | 0.791 | 0.980 | 0.983 | 0.928 |
| H2 | units | off | 0.886 | 1.000 | 0.763 | 0.980 | 0.933 | 0.926 |
| M1 | units | off | 0.910 | 0.989 | 0.876 | 0.940 | 0.900 | 0.889 |
| Q1 | units | off | 0.945 | 0.978 | 0.932 | 0.920 | 0.967 | 0.933 |
| Q2 | units | off | 0.827 | 0.935 | 0.757 | 0.900 | 0.817 | 0.821 |
| F0 | units | off | 0.907 | 0.989 | 0.802 | 0.980 | 0.983 | 0.939 |
| F1 | units | off | 0.898 | 0.989 | 0.791 | 0.940 | 0.983 | 0.933 |
| T1 | units | off | 0.901 | 0.989 | 0.791 | 0.980 | 0.983 | 0.928 |
| X1 | units | off | 0.912 | 1.000 | 0.825 | 1.000 | 0.967 | 0.908 |
| X2 | units | off | 0.987 | 1.000 | 0.977 | 1.000 | 1.000 | 0.978 |
| T4 8k | units | on | 0.957 | 1.000 | 0.932 | 1.000 | 0.950 | 0.944 |
| L1 | windows | off | 0.790 | 0.978 | 0.599 | 0.963 | 0.850 | 0.835 |
| S1 | windows | off | 0.759 | 0.978 | 0.588 | 0.930 | 0.800 | 0.752 |
| D1 | windows | off | 0.889 | 0.978 | 0.864 | 0.933 | 0.883 | 0.824 |
| C1 | windows | off | 0.912 | 1.000 | 0.898 | 0.970 | 0.850 | 0.860 |
| H1 | windows | off | 0.886 | 1.000 | 0.791 | 0.967 | 0.933 | 0.882 |
| H2 | windows | off | 0.862 | 0.989 | 0.746 | 0.960 | 0.900 | 0.880 |
| M1 | windows | off | 0.915 | 1.000 | 0.904 | 0.970 | 0.850 | 0.860 |
| Q1 | windows | off | 0.929 | 0.989 | 0.921 | 0.903 | 0.950 | 0.885 |
| Q2 | windows | off | 0.814 | 0.924 | 0.751 | 0.843 | 0.817 | 0.806 |
| F0 | windows | off | 0.900 | 1.000 | 0.819 | 0.967 | 0.950 | 0.888 |
| F1 | windows | off | 0.891 | 1.000 | 0.808 | 0.927 | 0.950 | 0.882 |
| T1 | windows | off | 0.886 | 1.000 | 0.791 | 0.967 | 0.933 | 0.882 |
| X1 | windows | off | 0.930 | 1.000 | 0.881 | 0.997 | 0.950 | 0.902 |
| X2 | windows | off | 0.981 | 1.000 | 0.972 | 1.000 | 1.000 | 0.957 |
| T4 8k | windows | on | 0.959 | 1.000 | 0.960 | 1.000 | 0.933 | 0.910 |

What each family shows:

- **`set1`:** everything except Q2 and S1 is between 0.81 and 0.98 Hit@1. This family cannot rank the methods, which is why run 1 was inconclusive.
- **`paraphrase`:** the family that separates the methods. BM25 and learned sparse lose more than half their Hit@1. Dense and ColBERT lose a quarter to a third. Only X2 (0.81–0.86) and T4 (0.89–0.93) stay high. H1 (0.49–0.50) is below dense (0.58–0.61).
- **`section_id`:** the easiest of the new families. H1, the rerankers and the agent reach 0.80 to 0.98. HyDE (0.72–0.76) and keyword expansion (0.52–0.54) do worst, because the generated text dilutes the id.
- **`twin_topic`:** the hardest family. Every ranking method without engine-type knowledge is at 0.28 to 0.55 Hit@1, rerankers included. Only the filter (0.68–0.70) and the agent (0.78–0.82) do better. The evidence is still retrieved: X2's Evidence@10 is 1.000. The problem is the order, not the recall.
- **`multi_section`:** Hit@1 is high because either of the two gold sections counts. The honest number is evidence: X2 without the gate reaches 0.96–0.98, T4 0.91–0.94, H1 0.88–0.93.

### The paraphrase family in detail (exploratory)

Each set 1 original is shown next to its two rewrites. Only the wording changes between the three columns.

| Method | Index | Gate | Hit@1 original (89) | Hit@1 user voice (88) | Hit@1 heavy (89) | Evidence@10 original | Evidence@10 user voice | Evidence@10 heavy |
|---|---|---|---|---|---|---|---|---|
| L1 | units | off | 0.854 | 0.557 | 0.270 | 0.978 | 0.807 | 0.438 |
| S1 | units | off | 0.775 | 0.477 | 0.202 | 0.989 | 0.841 | 0.427 |
| D1 | units | off | 0.831 | 0.659 | 0.494 | 0.966 | 0.909 | 0.843 |
| C1 | units | off | 0.820 | 0.716 | 0.528 | 0.989 | 0.909 | 0.831 |
| H1 | units | off | 0.899 | 0.648 | 0.348 | 0.989 | 0.898 | 0.685 |
| Q1 | units | off | 0.831 | 0.727 | 0.573 | 0.978 | 0.920 | 0.944 |
| Q2 | units | off | 0.697 | 0.489 | 0.427 | 0.933 | 0.818 | 0.697 |
| X1 | units | off | 0.921 | 0.761 | 0.315 | 1.000 | 0.943 | 0.708 |
| X2 | units | off | 0.944 | 0.886 | 0.742 | 1.000 | 1.000 | 0.955 |
| T4 8k | units | on | 0.989 | 0.920 | 0.865 | 1.000 | 0.955 | 0.910 |
| L1 | windows | off | 0.876 | 0.523 | 0.258 | 0.978 | 0.807 | 0.393 |
| S1 | windows | off | 0.820 | 0.511 | 0.180 | 0.978 | 0.807 | 0.371 |
| D1 | windows | off | 0.843 | 0.670 | 0.551 | 0.978 | 0.920 | 0.809 |
| C1 | windows | off | 0.899 | 0.727 | 0.517 | 1.000 | 0.955 | 0.843 |
| H1 | windows | off | 0.865 | 0.648 | 0.326 | 1.000 | 0.909 | 0.674 |
| Q1 | windows | off | 0.831 | 0.727 | 0.618 | 0.989 | 0.943 | 0.899 |
| Q2 | windows | off | 0.742 | 0.511 | 0.438 | 0.921 | 0.807 | 0.697 |
| X1 | windows | off | 0.978 | 0.852 | 0.517 | 1.000 | 0.966 | 0.798 |
| X2 | windows | off | 0.955 | 0.909 | 0.809 | 1.000 | 1.000 | 0.944 |
| T4 8k | windows | on | 0.966 | 0.932 | 0.933 | 1.000 | 0.977 | 0.944 |

- Heavy rewrites share 22% of their content stems with the gold section. BM25 finds the gold section first for about a quarter of them, and has the evidence in the top 10 for about 40%.
- **Fusion hurts here.** H1's Hit@1 on heavy rewrites (0.33–0.35) is far below dense alone (0.49–0.55), because RRF gives BM25's wrong hits the same weight as dense's right ones.
- **bge-reranker-v2-m3 does not recover them.** X1 reaches 0.315 on units and 0.517 on windows. jina-reranker-v3 reaches 0.742 and 0.809 from the same candidates.
- HyDE is the best first stage on heavy rewrites for evidence (0.90–0.94 in the top 10), ahead of dense (0.81–0.84).
- The agent is the least sensitive to wording: 0.87–0.93 on heavy rewrites.

## 4. RQ1: Which single retrieval signal works best?

**Answer:** dense and ColBERT are significantly better than BM25 on every metric shown, on both indexes and at both gate settings. Dense adds +0.072 Hit@1 and about +0.10 Evidence@10; ColBERT adds +0.085 to +0.102 Hit@1 and +0.10 to +0.12 Evidence@10. Learned sparse (S1) is not better than BM25: it is nominally worse everywhere, and significantly worse on MRR on units.

**Where it comes from:** only the paraphrase family. There dense and ColBERT gain +0.16 to +0.23 Hit@1 over BM25. On `set1`, `section_id`, `twin_topic` and `multi_section` no Hit@1 comparison between single signals is significant, and on `multi_section` BM25 is nominally ahead of dense.

**Practical reading:** BM25 alone is only safe when users quote the regulation. Dense gets most of ColBERT's gain at 0.02 s instead of 0.8 s per query. Dense against ColBERT was not a pre-declared comparison; ColBERT is nominally ahead by 0.01 to 0.03 Hit@1.

| Method | Index | Gate | Hit@1 | MRR@10 | nDCG@10 (95% CI) | Evidence@10 | Context tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| L1 | units | off | 0.586 | 0.684 | 0.662 (0.630–0.692) | 0.810 | 16,957 | 0.02 |
| S1 | units | off | 0.539 | 0.651 | 0.642 (0.611–0.673) | 0.792 | 17,568 | 0.02 |
| D1 | units | off | 0.659 | 0.765 | 0.749 (0.723–0.773) | 0.912 | 14,190 | 0.02 |
| C1 | units | off | 0.672 | 0.772 | 0.752 (0.727–0.777) | 0.908 | 11,643 | 0.81 |
| L1 | windows | off | 0.595 | 0.689 | 0.665 (0.635–0.696) | 0.790 | 8,217 | 0.02 |
| S1 | windows | off | 0.561 | 0.665 | 0.653 (0.622–0.684) | 0.759 | 8,138 | 0.02 |
| D1 | windows | off | 0.667 | 0.777 | 0.755 (0.731–0.779) | 0.889 | 7,764 | 0.02 |
| C1 | windows | off | 0.697 | 0.798 | 0.767 (0.742–0.790) | 0.912 | 7,455 | 0.77 |

| Comparison | Index | Gate | ΔHit@1 | ΔMRR@10 | ΔnDCG@10 | ΔEvidence@10 |
|---|---|---|---|---|---|---|
| S1 vs L1 | units | off | -0.047 (p=0.075) | **-0.033 (p=0.029)** | -0.020 (p=0.083) | -0.018 (p=0.269) |
| S1 vs L1 | windows | off | -0.034 (p=0.217) | -0.024 (p=0.113) | -0.012 (p=0.253) | -0.030 (p=0.122) |
| D1 vs L1 | units | off | **+0.072 (p=0.022)** | **+0.081 (p=0.001)** | **+0.087 (p<0.001)** | **+0.103 (p<0.001)** |
| D1 vs L1 | windows | off | **+0.072 (p=0.013)** | **+0.088 (p<0.001)** | **+0.090 (p<0.001)** | **+0.099 (p<0.001)** |
| C1 vs L1 | units | off | **+0.085 (p=0.005)** | **+0.088 (p<0.001)** | **+0.091 (p<0.001)** | **+0.099 (p<0.001)** |
| C1 vs L1 | windows | off | **+0.102 (p<0.001)** | **+0.109 (p<0.001)** | **+0.102 (p<0.001)** | **+0.123 (p<0.001)** |

The gate-on tests lead to the same conclusion (D1 and C1 significant on all four metrics; S1 additionally loses Evidence@10 on units, −0.040, p = 0.031).

By family, ΔHit@1 (exploratory):

| Comparison | Index | Gate | set1 | paraphrase | section_id | twin_topic | multi_section |
|---|---|---|---|---|---|---|---|
| S1 vs L1 | units | off | -0.076 (p=1.000) | -0.073 (p=0.104) | +0.120 (p=0.660) | +0.067 (p=1.000) | -0.133 (p=0.095) |
| S1 vs L1 | windows | off | -0.054 (p=1.000) | -0.045 (p=0.431) | +0.140 (p=0.262) | +0.033 (p=1.000) | -0.133 (p=0.084) |
| D1 vs L1 | units | off | -0.011 (p=1.000) | **+0.164 (p=0.003)** | +0.060 (p=1.000) | +0.183 (p=0.119) | -0.089 (p=0.452) |
| D1 vs L1 | windows | off | -0.022 (p=1.000) | **+0.220 (p<0.001)** | +0.060 (p=1.000) | +0.067 (p=1.000) | -0.111 (p=0.170) |
| C1 vs L1 | units | off | -0.022 (p=1.000) | **+0.209 (p<0.001)** | +0.040 (p=1.000) | +0.067 (p=1.000) | -0.011 (p=1.000) |
| C1 vs L1 | windows | off | +0.033 (p=1.000) | **+0.232 (p<0.001)** | +0.160 (p=0.262) | +0.000 (p=1.000) | -0.044 (p=1.000) |

## 5. RQ2: Does hybrid fusion beat the single signals?

**Answer:** H1 beats BM25 on every metric (Hit@1 +0.05 to +0.09, Evidence@10 +0.09 to +0.15). It does **not** beat dense with the gate off: no metric differs significantly on either index. With the gate on, H1 has significantly more evidence than dense (+0.05 Evidence@10), but that is a side effect of list length: a gated dense list keeps about 6.5 hits and the gated fused list about 10. H2 (adding learned sparse) and M1 (ColBERT reranking of dense) add nothing; H2 is slightly worse on units, and M1 loses evidence once its second gate is on.

**By family:** H1 is nominally ahead of dense where exact terms matter (`section_id`, `multi_section`, `twin_topic`; significant on units for the first two) and behind it on paraphrases, where its MRR is 0.10 to 0.12 lower. So the hybrid is not a free improvement over dense. It trades wording robustness for term matching.

**Practical reading:** in run 1 H1 looked like the safe first stage because its recall was 1.000. On run 2 its Evidence@10 is 0.89–0.90 (gate off), the same as dense. What still speaks for H1 is the depth of its candidate list: X2 reranking H1's top 50 reaches 0.98–0.99 evidence, so the evidence is in the fused top 50 even when it is not in the fused top 10. H1 is a good candidate generator and a mediocre final ranking.

| Method | Index | Gate | Hit@1 | MRR@10 | nDCG@10 (95% CI) | Evidence@10 | Context tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| L1 | units | off | 0.586 | 0.684 | 0.662 (0.630–0.692) | 0.810 | 16,957 | 0.02 |
| D1 | units | off | 0.659 | 0.765 | 0.749 (0.723–0.773) | 0.912 | 14,190 | 0.02 |
| H1 | units | off | 0.680 | 0.767 | 0.745 (0.718–0.771) | 0.901 | 16,052 | 0.02 |
| H2 | units | off | 0.652 | 0.745 | 0.728 (0.700–0.756) | 0.886 | 16,853 | 0.02 |
| M1 | units | off | 0.672 | 0.772 | 0.752 (0.727–0.776) | 0.910 | 12,004 | 0.14 |
| L1 | windows | off | 0.595 | 0.689 | 0.665 (0.635–0.696) | 0.790 | 8,217 | 0.02 |
| D1 | windows | off | 0.667 | 0.777 | 0.755 (0.731–0.779) | 0.889 | 7,764 | 0.02 |
| H1 | windows | off | 0.646 | 0.752 | 0.739 (0.713–0.766) | 0.886 | 7,983 | 0.02 |
| H2 | windows | off | 0.648 | 0.749 | 0.735 (0.708–0.762) | 0.862 | 8,044 | 0.02 |
| M1 | windows | off | 0.697 | 0.799 | 0.767 (0.743–0.791) | 0.915 | 7,524 | 0.15 |

| Comparison | Index | Gate | ΔHit@1 | ΔMRR@10 | ΔnDCG@10 | ΔEvidence@10 |
|---|---|---|---|---|---|---|
| H1 vs L1 | units | off | **+0.094 (p<0.001)** | **+0.083 (p<0.001)** | **+0.083 (p<0.001)** | **+0.091 (p<0.001)** |
| H1 vs L1 | units | on | **+0.092 (p<0.001)** | **+0.103 (p<0.001)** | **+0.113 (p<0.001)** | **+0.149 (p<0.001)** |
| H1 vs L1 | windows | off | **+0.051 (p=0.050)** | **+0.063 (p<0.001)** | **+0.074 (p<0.001)** | **+0.097 (p<0.001)** |
| H1 vs L1 | windows | on | **+0.051 (p=0.050)** | **+0.085 (p<0.001)** | **+0.107 (p<0.001)** | **+0.151 (p<0.001)** |
| H1 vs D1 | units | off | +0.021 (p=1.000) | +0.002 (p=1.000) | -0.004 (p=1.000) | -0.011 (p=0.856) |
| H1 vs D1 | units | on | +0.019 (p=1.000) | +0.023 (p=0.280) | **+0.029 (p=0.010)** | **+0.049 (p<0.001)** |
| H1 vs D1 | windows | off | -0.021 (p=1.000) | -0.025 (p=0.412) | -0.016 (p=0.583) | -0.002 (p=1.000) |
| H1 vs D1 | windows | on | -0.021 (p=1.000) | -0.002 (p=1.000) | +0.022 (p=0.054) | **+0.054 (p<0.001)** |
| H2 vs H1 | units | off | -0.028 (p=0.454) | -0.022 (p=0.058) | **-0.016 (p=0.015)** | -0.015 (p=0.378) |
| H2 vs H1 | units | on | -0.017 (p=1.000) | -0.019 (p=0.145) | -0.009 (p=0.297) | -0.015 (p=0.192) |
| H2 vs H1 | windows | off | +0.002 (p=1.000) | -0.003 (p=1.000) | -0.004 (p=0.927) | -0.025 (p=0.065) |
| H2 vs H1 | windows | on | +0.009 (p=1.000) | -0.004 (p=1.000) | +0.001 (p=0.927) | -0.001 (p=1.000) |
| M1 vs D1 | units | off | +0.013 (p=1.000) | +0.007 (p=1.000) | +0.004 (p=1.000) | -0.002 (p=0.907) |
| M1 vs D1 | units | on | +0.011 (p=1.000) | +0.004 (p=1.000) | -0.006 (p=1.000) | **-0.027 (p<0.001)** |
| M1 vs D1 | windows | off | +0.030 (p=0.922) | +0.022 (p=0.412) | +0.012 (p=0.583) | +0.026 (p=0.080) |
| M1 vs D1 | windows | on | +0.026 (p=1.000) | +0.013 (p=1.000) | -0.006 (p=0.894) | **-0.022 (p=0.006)** |

By family, ΔMRR@10 (exploratory):

| Comparison | Index | Gate | set1 | paraphrase | section_id | twin_topic | multi_section |
|---|---|---|---|---|---|---|---|
| H1 vs L1 | units | off | +0.034 (p=0.526) | **+0.100 (p<0.001)** | +0.080 (p=0.267) | **+0.175 (p<0.001)** | +0.038 (p=0.509) |
| H1 vs L1 | windows | off | +0.006 (p=1.000) | **+0.116 (p<0.001)** | +0.086 (p=0.148) | **+0.085 (p=0.036)** | -0.011 (p=1.000) |
| H1 vs D1 | units | off | +0.046 (p=0.449) | **-0.095 (p=0.027)** | **+0.081 (p=0.046)** | +0.039 (p=0.372) | **+0.080 (p=0.015)** |
| H1 vs D1 | windows | off | +0.015 (p=1.000) | **-0.120 (p=0.001)** | +0.053 (p=0.459) | +0.030 (p=1.000) | +0.040 (p=0.374) |

## 6. RQ3: Does a reranker improve early precision, and how do the two compare?

**Answer:** yes, and the reranker matters more than any other choice in the non-LLM pipeline.

- **X2 (jina-reranker-v3) against H1:** +0.158 Hit@1 on units and +0.207 on windows with the gate off, with +0.09 Evidence@10. Significant on every ranking metric in all four settings.
- **X1 (bge-reranker-v2-m3) against H1:** significant on windows only (+0.128 Hit@1). On units it changes nothing (+0.015, p = 0.552), and with the gate on it loses evidence (−0.046, p < 0.001), because its output gate cuts gold hits.
- **X1 against X2:** X2 is significantly better on every metric in all four settings: −0.14 Hit@1 for X1 on units, −0.08 on windows.

**Where it comes from:** paraphrases again. On `set1` the two rerankers are tied, exactly as in run 1. On paraphrases X2 leads X1 by 0.28 (units) and 0.18 (windows) Hit@1. Neither reranker helps on `twin_topic`: they cannot tell the piston section from the turbine one (section 8).

**Gate:** with the gate on, X2 keeps its Hit@1 (0.844–0.846) but its evidence falls to the level of gated H1 (0.92–0.95), because the reranker only sees about 10 candidates. Section 11 covers this.

**Practical reading:** jina-reranker-v3 is the single most useful component in run 2. Its licence is CC BY-NC 4.0 (non-commercial), which fits this research project. Latency: X2 over 50 candidates takes 8.2 s on windows and 25.1 s on units; with the gate on, 0.7 s and 1.6 s.

| Method | Index | Gate | Hit@1 | MRR@10 | nDCG@10 (95% CI) | Evidence@10 | Context tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| H1 | units | off | 0.680 | 0.767 | 0.745 (0.718–0.771) | 0.901 | 16,052 | 0.02 |
| H1 | units | on | 0.678 | 0.784 | 0.755 (0.730–0.779) | 0.937 | 7,353 | 0.02 |
| X1 | units | off | 0.695 | 0.788 | 0.747 (0.721–0.772) | 0.912 | 20,136 | 6.99 |
| X1 | units | on | 0.708 | 0.797 | 0.744 (0.717–0.770) | 0.891 | 5,183 | 1.41 |
| X2 | units | off | 0.838 | 0.907 | 0.864 (0.847–0.880) | 0.987 | 13,760 | 25.15 |
| X2 | units | on | 0.844 | 0.899 | 0.831 (0.810–0.852) | 0.945 | 4,698 | 1.58 |
| H1 | windows | off | 0.646 | 0.752 | 0.739 (0.713–0.766) | 0.886 | 7,983 | 0.02 |
| H1 | windows | on | 0.646 | 0.769 | 0.743 (0.718–0.767) | 0.909 | 4,042 | 0.02 |
| X1 | windows | off | 0.774 | 0.846 | 0.796 (0.773–0.818) | 0.930 | 8,328 | 2.40 |
| X1 | windows | on | 0.768 | 0.837 | 0.767 (0.741–0.791) | 0.897 | 2,551 | 0.52 |
| X2 | windows | off | 0.853 | 0.912 | 0.864 (0.847–0.881) | 0.981 | 7,635 | 8.23 |
| X2 | windows | on | 0.846 | 0.895 | 0.814 (0.792–0.836) | 0.920 | 2,460 | 0.71 |

| Comparison | Index | Gate | ΔHit@1 | ΔMRR@10 | ΔnDCG@10 | ΔEvidence@10 |
|---|---|---|---|---|---|---|
| X1 vs H1 | units | off | +0.015 (p=0.552) | +0.021 (p=0.274) | +0.002 (p=0.821) | +0.011 (p=0.373) |
| X1 vs H1 | units | on | +0.030 (p=0.394) | +0.013 (p=0.322) | -0.011 (p=0.422) | **-0.046 (p<0.001)** |
| X1 vs H1 | windows | off | **+0.128 (p<0.001)** | **+0.094 (p<0.001)** | **+0.057 (p<0.001)** | **+0.043 (p=0.002)** |
| X1 vs H1 | windows | on | **+0.122 (p<0.001)** | **+0.068 (p<0.001)** | **+0.023 (p=0.002)** | -0.012 (p=0.135) |
| X2 vs H1 | units | off | **+0.158 (p<0.001)** | **+0.140 (p<0.001)** | **+0.119 (p<0.001)** | **+0.086 (p<0.001)** |
| X2 vs H1 | units | on | **+0.166 (p<0.001)** | **+0.115 (p<0.001)** | **+0.077 (p<0.001)** | +0.009 (p=0.186) |
| X2 vs H1 | windows | off | **+0.207 (p<0.001)** | **+0.160 (p<0.001)** | **+0.125 (p<0.001)** | **+0.095 (p<0.001)** |
| X2 vs H1 | windows | on | **+0.200 (p<0.001)** | **+0.127 (p<0.001)** | **+0.071 (p<0.001)** | +0.011 (p=0.135) |
| X1 vs X2 | units | off | **-0.143 (p<0.001)** | **-0.120 (p<0.001)** | **-0.117 (p<0.001)** | **-0.075 (p<0.001)** |
| X1 vs X2 | units | on | **-0.136 (p<0.001)** | **-0.102 (p<0.001)** | **-0.087 (p<0.001)** | **-0.054 (p<0.001)** |
| X1 vs X2 | windows | off | **-0.079 (p<0.001)** | **-0.066 (p<0.001)** | **-0.068 (p<0.001)** | **-0.052 (p<0.001)** |
| X1 vs X2 | windows | on | **-0.079 (p<0.001)** | **-0.059 (p<0.001)** | **-0.048 (p<0.001)** | **-0.023 (p=0.003)** |

By family, ΔHit@1, gate off (exploratory):

| Comparison | Index | Gate | set1 | paraphrase | section_id | twin_topic | multi_section |
|---|---|---|---|---|---|---|---|
| X1 vs H1 | units | off | +0.011 (p=1.000) | +0.040 (p=0.388) | +0.060 (p=1.000) | -0.033 (p=1.000) | -0.022 (p=1.000) |
| X1 vs H1 | windows | off | **+0.120 (p=0.006)** | **+0.198 (p<0.001)** | +0.100 (p=0.259) | +0.050 (p=1.000) | +0.067 (p=0.628) |
| X2 vs H1 | units | off | +0.054 (p=1.000) | **+0.316 (p<0.001)** | +0.080 (p=0.754) | +0.033 (p=1.000) | +0.078 (p=0.279) |
| X2 vs H1 | windows | off | +0.098 (p=0.067) | **+0.373 (p<0.001)** | +0.140 (p=0.085) | +0.067 (p=1.000) | **+0.122 (p=0.018)** |
| X1 vs X2 | units | off | -0.043 (p=1.000) | **-0.277 (p<0.001)** | -0.020 (p=1.000) | -0.067 (p=1.000) | -0.100 (p=0.183) |
| X1 vs X2 | windows | off | +0.022 (p=1.000) | **-0.175 (p<0.001)** | -0.040 (p=0.502) | -0.017 (p=1.000) | -0.056 (p=0.628) |

Latency note: X1's gate-on figures are a pro-rata share of the gate-off batch, because its scores are cached per (query, unit). X2's gate-on figures are measured directly.

## 7. RQ4: Do LLM query rewrites help?

**HyDE (Q1): yes, a little.** Against plain dense it gains +0.035 to +0.041 MRR, +0.03 nDCG and +0.03 to +0.04 Evidence@10, all significant with the gate off. Hit@1 rises by 0.041 on both indexes but is not significant (p = 0.083 and 0.102). The gain is on `paraphrase` and `multi_section`. On heavy paraphrases HyDE lifts Evidence@10 from 0.81–0.84 to 0.90–0.94. It is nominally worse on `section_id`, where the invented passage dilutes the id.

**Keyword expansion (Q2): no.** It is significantly worse than BM25 on units (Hit@1 −0.068, MRR −0.047) and not significantly different on windows. By family it does what it was meant to do on paraphrases (MRR +0.07 to +0.10, Evidence@10 +0.14 to +0.15) and damages everything else (MRR −0.09 to −0.18 on `set1`, `section_id` and `multi_section`). Even on paraphrases, BM25 with keywords (Hit@1 0.46–0.48) stays below plain dense (0.58–0.61).

**Practical reading:** RQ4 asked whether rewrites help "especially when the question's wording differs from the regulation's". Run 2 can answer that, and the answer is yes for HyDE and for wording-gap questions only. But the gain is small next to a reranker: X2 on windows is 0.145 Hit@1 above Q1, and HyDE costs 13 s per query on Qwen. HyDE followed by a reranker was not tested.

| Method | Index | Gate | Hit@1 | MRR@10 | nDCG@10 (95% CI) | Evidence@10 | Context tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| L1 | units | off | 0.586 | 0.684 | 0.662 (0.630–0.692) | 0.810 | 16,957 | 0.02 |
| Q2 | units | off | 0.518 | 0.638 | 0.633 (0.604–0.662) | 0.827 | 18,541 | 4.77 |
| D1 | units | off | 0.659 | 0.765 | 0.749 (0.723–0.773) | 0.912 | 14,190 | 0.02 |
| Q1 | units | off | 0.699 | 0.805 | 0.780 (0.757–0.802) | 0.945 | 12,953 | 12.98 |
| L1 | windows | off | 0.595 | 0.689 | 0.665 (0.635–0.696) | 0.790 | 8,217 | 0.02 |
| Q2 | windows | off | 0.557 | 0.670 | 0.657 (0.629–0.685) | 0.814 | 8,447 | 4.77 |
| D1 | windows | off | 0.667 | 0.777 | 0.755 (0.731–0.779) | 0.889 | 7,764 | 0.02 |
| Q1 | windows | off | 0.708 | 0.811 | 0.782 (0.760–0.802) | 0.929 | 7,632 | 12.98 |

| Comparison | Index | Gate | ΔHit@1 | ΔMRR@10 | ΔnDCG@10 | ΔEvidence@10 |
|---|---|---|---|---|---|---|
| Q1 vs D1 | units | off | +0.041 (p=0.083) | **+0.041 (p=0.001)** | **+0.031 (p<0.001)** | **+0.032 (p=0.009)** |
| Q1 vs D1 | windows | off | +0.041 (p=0.102) | **+0.035 (p=0.003)** | **+0.026 (p<0.001)** | **+0.041 (p<0.001)** |
| Q2 vs L1 | units | off | **-0.068 (p=0.012)** | **-0.047 (p=0.005)** | **-0.028 (p=0.034)** | +0.017 (p=0.310) |
| Q2 vs L1 | windows | off | -0.038 (p=0.169) | -0.019 (p=0.222) | -0.008 (p=0.546) | +0.024 (p=0.349) |

By family, ΔMRR@10 (exploratory):

| Comparison | Index | Gate | set1 | paraphrase | section_id | twin_topic | multi_section |
|---|---|---|---|---|---|---|---|
| Q1 vs D1 | units | off | +0.009 (p=1.000) | **+0.068 (p=0.003)** | -0.026 (p=0.977) | +0.029 (p=0.823) | **+0.063 (p=0.032)** |
| Q1 vs D1 | windows | off | -0.000 (p=1.000) | **+0.057 (p=0.010)** | -0.028 (p=0.955) | +0.073 (p=0.066) | +0.036 (p=0.163) |
| Q2 vs L1 | units | off | **-0.123 (p<0.001)** | **+0.069 (p=0.033)** | **-0.180 (p=0.001)** | -0.067 (p=0.691) | **-0.108 (p=0.002)** |
| Q2 vs L1 | windows | off | **-0.095 (p<0.001)** | **+0.102 (p=0.004)** | **-0.132 (p=0.004)** | -0.055 (p=0.532) | **-0.094 (p<0.001)** |

By family, ΔEvidence@10 (exploratory):

| Comparison | Index | Gate | set1 | paraphrase | section_id | twin_topic | multi_section |
|---|---|---|---|---|---|---|---|
| Q1 vs D1 | units | off | +0.011 (p=1.000) | **+0.056 (p=0.040)** | -0.040 (p=1.000) | +0.033 (p=1.000) | **+0.046 (p=0.042)** |
| Q1 vs D1 | windows | off | +0.011 (p=1.000) | **+0.056 (p=0.046)** | -0.030 (p=1.000) | +0.067 (p=0.481) | **+0.061 (p=0.037)** |
| Q2 vs L1 | units | off | -0.043 (p=0.368) | **+0.136 (p<0.001)** | -0.080 (p=0.376) | -0.100 (p=0.214) | -0.020 (p=0.977) |
| Q2 vs L1 | windows | off | -0.054 (p=0.181) | **+0.153 (p<0.001)** | **-0.120 (p=0.022)** | -0.033 (p=0.723) | -0.030 (p=0.236) |

## 8. RQ5: Do metadata and document structure help?

### Engine-type filter (F1)

**Answer:** yes. Pooled Hit@1 rises by 0.023 to 0.045 over H1 and is significant in three of the four settings; nDCG is significant in all four. Evidence recall does not change: the filter reorders, it does not find more.

**Where it comes from:** `twin_topic` only. There Hit@1 rises by 0.18 (units) and 0.23 (windows). The table shows why: without the filter, H1 puts a section of the **other** engine type at rank 1 on 27–38% of twin questions, and the rerankers do so on 38–43%. With the filter that never happens.

| Method | Index | Gate | Hit@1 | Hit@1 explicit cue (30) | Hit@1 implied cue (30) | Hit@3 | Evidence@5 | Rank 1 is a section of the other engine type |
|---|---|---|---|---|---|---|---|---|
| L1 | units | off | 0.300 | 0.433 | 0.167 | 0.700 | 0.833 | 0.250 |
| D1 | units | off | 0.483 | 0.600 | 0.367 | 0.817 | 0.867 | 0.383 |
| H1 | units | off | 0.517 | 0.600 | 0.433 | 0.833 | 0.917 | 0.267 |
| F0 | units | off | 0.700 | 0.800 | 0.600 | 0.900 | 0.967 | 0.000 |
| F1 | units | off | 0.700 | 0.800 | 0.600 | 0.917 | 0.967 | 0.000 |
| X1 | units | off | 0.483 | 0.567 | 0.400 | 0.833 | 0.867 | 0.400 |
| X2 | units | off | 0.550 | 0.600 | 0.500 | 0.983 | 1.000 | 0.400 |
| T4 8k | units | on | 0.817 | 0.800 | 0.833 | 0.933 | 0.950 | 0.167 |
| L1 | windows | off | 0.367 | 0.500 | 0.233 | 0.750 | 0.767 | 0.267 |
| D1 | windows | off | 0.433 | 0.533 | 0.333 | 0.817 | 0.833 | 0.433 |
| H1 | windows | off | 0.450 | 0.567 | 0.333 | 0.850 | 0.833 | 0.383 |
| F0 | windows | off | 0.683 | 0.800 | 0.567 | 0.900 | 0.917 | 0.000 |
| F1 | windows | off | 0.683 | 0.800 | 0.567 | 0.917 | 0.917 | 0.000 |
| X1 | windows | off | 0.500 | 0.600 | 0.400 | 0.850 | 0.917 | 0.383 |
| X2 | windows | off | 0.517 | 0.567 | 0.467 | 0.950 | 0.967 | 0.433 |
| T4 8k | windows | on | 0.783 | 0.767 | 0.800 | 0.917 | 0.900 | 0.183 |

- F1 equals the oracle F0 on twin questions, so the LLM extraction is not the limit. Qwen returned an engine type for all 60 twin questions and none excluded the gold.
- Over all 469 scored questions F1's extraction excluded the gold section 4 times (2 heavy paraphrases and 2 `section_id` questions; none of the four states an engine type). That is why F1 is nominally below H1 on `section_id`.
- The filter still leaves twin Hit@1 at 0.68–0.70. The remaining rank-1 errors are sections of the right engine type or of Subpart A.
- The agent reaches 0.78–0.82 without any filter, and is the only method that handles an implied engine type as well as an explicit one.

### Section-id lookup (T1)

**Answer:** no measurable benefit. Hit@1 changes by +0.004 to +0.009 (not significant). nDCG rises by 0.004, which is significant and irrelevant.

Run 1 could not test T1 because only 6 questions named an id. Run 2 has 68 such questions, and the reason for the null result is visible in the `section_id` family:

| Method | Index | Gate | canonical (19) | paragraph (15) | loose (16) | lookup (20) | objective (10) | redirect (10) | summary (10) |
|---|---|---|---|---|---|---|---|---|---|
| L1 | units | off | 0.579 | 0.933 | 0.750 | 0.950 | 0.700 | 0.900 | 0.200 |
| D1 | units | off | 0.632 | 1.000 | 0.812 | 1.000 | 1.000 | 0.700 | 0.300 |
| H1 | units | off | 0.737 | 1.000 | 0.938 | 1.000 | 1.000 | 0.800 | 0.600 |
| T1 | units | off | 0.684 | 1.000 | 0.938 | 1.000 | 1.000 | 0.400 | 0.900 |
| X1 | units | off | 0.947 | 1.000 | 0.875 | 1.000 | 1.000 | 1.000 | 0.700 |
| X2 | units | off | 0.895 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 |
| T4 8k | units | on | 0.947 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 1.000 |
| L1 | windows | off | 0.579 | 0.867 | 0.625 | 0.900 | 0.800 | 0.800 | 0.000 |
| D1 | windows | off | 0.684 | 0.800 | 0.750 | 0.850 | 1.000 | 0.700 | 0.300 |
| H1 | windows | off | 0.684 | 0.867 | 0.875 | 0.900 | 1.000 | 0.800 | 0.400 |
| T1 | windows | off | 0.684 | 1.000 | 0.875 | 1.000 | 1.000 | 0.400 | 0.800 |
| X1 | windows | off | 0.842 | 1.000 | 0.875 | 1.000 | 1.000 | 1.000 | 0.500 |
| X2 | windows | off | 0.895 | 1.000 | 0.938 | 1.000 | 1.000 | 1.000 | 0.700 |
| T4 8k | windows | on | 0.947 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 1.000 |

- T1 puts the named section first. That is right for lookups and objectives, where H1 was already at 0.90–1.00.
- It **fixes summaries** (H1 0.40–0.60, T1 0.80–0.90): a summary question has no content words for H1 to match.
- It **breaks redirects** (H1 0.80, T1 0.40): the user names the CS, but the answer is in its AMC.
- It misses loosely written ids ("cse 730") by design, so those rows equal H1.
- The two effects cancel in the family average.

**Practical reading:** F1 is worth its 1 s and its single short LLM call wherever the corpus has piston and turbine twins. T1 as written is not worth adding; X2 and T4 already reach 0.94–0.98 on this family. If id lookup is kept, it should add the named section's group (the CS and its AMCs), not move one section to rank 1.

| Method | Index | Gate | Hit@1 | MRR@10 | nDCG@10 (95% CI) | Evidence@10 | Context tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| H1 | units | off | 0.680 | 0.767 | 0.745 (0.718–0.771) | 0.901 | 16,052 | 0.02 |
| F0 | units | off | 0.710 | 0.789 | 0.768 (0.741–0.794) | 0.907 | 15,968 | 0.02 |
| F1 | units | off | 0.704 | 0.782 | 0.760 (0.733–0.787) | 0.898 | 15,950 | 0.97 |
| T1 | units | off | 0.684 | 0.770 | 0.748 (0.721–0.775) | 0.901 | 16,052 | 0.02 |
| H1 | windows | off | 0.646 | 0.752 | 0.739 (0.713–0.766) | 0.886 | 7,983 | 0.02 |
| F0 | windows | off | 0.689 | 0.782 | 0.771 (0.745–0.796) | 0.900 | 7,950 | 0.02 |
| F1 | windows | off | 0.687 | 0.778 | 0.763 (0.736–0.789) | 0.891 | 7,952 | 0.97 |
| T1 | windows | off | 0.655 | 0.756 | 0.743 (0.717–0.769) | 0.886 | 7,983 | 0.02 |

| Comparison | Index | Gate | ΔHit@1 | ΔMRR@10 | ΔnDCG@10 | ΔEvidence@10 |
|---|---|---|---|---|---|---|
| F1 vs H1 | units | off | +0.023 (p=0.108) | +0.015 (p=0.083) | **+0.015 (p=0.026)** | -0.003 (p=1.000) |
| F1 vs H1 | units | on | **+0.030 (p=0.040)** | +0.017 (p=0.057) | **+0.016 (p=0.026)** | -0.004 (p=1.000) |
| F1 vs H1 | windows | off | **+0.041 (p=0.004)** | **+0.026 (p=0.001)** | **+0.024 (p<0.001)** | +0.004 (p=1.000) |
| F1 vs H1 | windows | on | **+0.045 (p=0.002)** | **+0.029 (p=0.001)** | **+0.026 (p<0.001)** | +0.003 (p=1.000) |
| T1 vs H1 | units | off | +0.004 (p=1.000) | +0.003 (p=0.634) | **+0.004 (p=0.046)** | +0.000 (p=1.000) |
| T1 vs H1 | units | on | +0.004 (p=1.000) | +0.005 (p=0.634) | **+0.004 (p=0.046)** | +0.000 (p=1.000) |
| T1 vs H1 | windows | off | +0.009 (p=0.755) | +0.004 (p=0.480) | **+0.004 (p=0.011)** | +0.000 (p=1.000) |
| T1 vs H1 | windows | on | +0.009 (p=0.755) | +0.005 (p=0.480) | **+0.004 (p=0.006)** | +0.003 (p=1.000) |

By family, ΔHit@1 (exploratory):

| Comparison | Index | Gate | set1 | paraphrase | section_id | twin_topic | multi_section |
|---|---|---|---|---|---|---|---|
| F1 vs H1 | units | off | +0.000 (p=1.000) | +0.000 (p=1.000) | -0.040 (p=1.000) | **+0.183 (p=0.012)** | +0.022 (p=1.000) |
| F1 vs H1 | windows | off | +0.022 (p=1.000) | +0.023 (p=0.842) | -0.040 (p=1.000) | **+0.233 (p=0.002)** | +0.011 (p=1.000) |
| T1 vs H1 | units | off | +0.000 (p=1.000) | +0.017 (p=1.000) | -0.020 (p=1.000) | +0.000 (p=1.000) | +0.000 (p=1.000) |
| T1 vs H1 | windows | off | +0.000 (p=1.000) | +0.011 (p=0.999) | +0.040 (p=1.000) | +0.000 (p=1.000) | +0.000 (p=1.000) |

F0 is reported as a bound and is not tested.

## 9. RQ6: Does expansion bring more evidence? Does an agent beat fixed rules?

Run 1 could not answer this, because every question had one gold section. Run 2 has 90 questions that need two.

### Fixed rules (T2, T3, T2+T3)

**Answer:** no. Across all 36 tests of evidence in context (three rules, three budgets, two gate settings, two indexes) no fixed rule significantly increases it, and 34 of the 36 differences are decreases. T3 and T2+T3 also significantly lower MRR in every one of those settings (−0.02 to −0.04), because the added sections push retrieved hits down or out of the budget.

**On multi-section questions specifically** (the 90 `multi_section` questions; exploratory):

| Method | Index | Gate | All-gold@5 | All-gold@10 | Supporting recall@10 | Complete evidence in context | …cs_amc (30) | …reference (40) | …comparison (20) | Context tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| H1 | units | off | 0.767 | 0.900 | 0.957 | 0.933 | 0.933 | 0.950 | 0.900 | 16,052 |
| X2 | units | off | 0.922 | 0.967 | 1.000 | 0.967 | 0.967 | 0.975 | 0.950 | 13,760 |
| T0 8k | units | on | 0.722 | 0.833 | 0.914 | 0.844 | 0.933 | 0.825 | 0.750 | 6,281 |
| T2 8k | units | on | 0.689 | 0.756 | 0.900 | 0.756 | 0.900 | 0.750 | 0.550 | 6,963 |
| T3 8k | units | on | 0.644 | 0.778 | 0.986 | 0.822 | 0.900 | 0.875 | 0.600 | 7,886 |
| T2+T3 8k | units | on | 0.633 | 0.756 | 0.986 | 0.800 | 0.867 | 0.875 | 0.550 | 7,905 |
| T4 8k | units | on | 0.800 | 0.911 | 0.929 | 0.778 | 0.933 | 0.625 | 0.850 | 3,977 |
| T3 16k | units | on | 0.633 | 0.811 | 0.986 | 0.867 | 0.933 | 0.900 | 0.700 | 15,276 |
| T4 16k | units | on | 0.811 | 0.911 | 0.929 | 0.789 | 0.933 | 0.625 | 0.900 | 4,359 |
| H1 | windows | off | 0.833 | 0.933 | 0.986 | 0.922 | 0.967 | 0.950 | 0.800 | 7,983 |
| X2 | windows | off | 0.944 | 0.967 | 1.000 | 0.944 | 0.967 | 0.950 | 0.900 | 7,635 |
| T0 8k | windows | on | 0.722 | 0.800 | 0.914 | 0.767 | 0.933 | 0.725 | 0.600 | 4,025 |
| T2 8k | windows | on | 0.700 | 0.756 | 0.900 | 0.744 | 0.900 | 0.725 | 0.550 | 6,408 |
| T3 8k | windows | on | 0.656 | 0.800 | 0.986 | 0.822 | 0.933 | 0.900 | 0.500 | 7,428 |
| T2+T3 8k | windows | on | 0.644 | 0.778 | 0.986 | 0.822 | 0.933 | 0.925 | 0.450 | 7,643 |
| T4 8k | windows | on | 0.867 | 0.944 | 0.971 | 0.789 | 0.967 | 0.675 | 0.750 | 2,434 |
| T3 16k | windows | on | 0.667 | 0.811 | 0.986 | 0.844 | 0.933 | 0.950 | 0.500 | 13,829 |
| T4 16k | windows | on | 0.844 | 0.922 | 0.957 | 0.778 | 0.967 | 0.625 | 0.800 | 2,702 |

- **T3 does what it was built for.** It raises supporting recall@10 from 0.914 to 0.986, and on `reference` questions it raises complete evidence from 0.825 to 0.875 on units and from 0.725 to 0.900 on windows (8k, gate on).
- **It pays for it elsewhere.** `comparison` questions get worse (the piston and turbine sections do not cite each other), and so do paraphrases. The net change on the whole `multi_section` family is not significant (+0.046 on windows, −0.014 on units).
- **T2 (parent expansion) only costs.** It fills the budget with the rest of sections that were already hit.
- **A better ranking does more than any expansion rule.** X2 without the gate has all gold sections in the top 5 for 92–94% of these questions, against 72% for T0 and 64–66% for T3 (8k, gate on).

### Agent (T4) against the fixed rules

**Answer:** the agent is better on every metric. Against T2+T3 at the same budget it gains +0.22 to +0.28 Hit@1 and +0.18 to +0.21 MRR in all six settings (all significant), and +0.03 to +0.07 evidence in context (significant in five of six).

**What the agent actually does** (8k budget):

| Index | Family | n | First action: done | expand | follow_refs | search | 3+ searches | LLM calls | LLM seconds | Context tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| units | set1 | 92 | 91 | 1 | 0 | 0 | 0 | 1.02 | 10.4 | 3,447 |
| units | paraphrase | 177 | 137 | 5 | 2 | 33 | 9 | 1.48 | 17.5 | 4,344 |
| units | section_id | 50 | 45 | 5 | 0 | 0 | 0 | 1.12 | 11.8 | 3,821 |
| units | twin_topic | 60 | 52 | 2 | 0 | 6 | 1 | 1.20 | 13.3 | 3,559 |
| units | multi_section | 90 | 75 | 4 | 2 | 9 | 1 | 1.30 | 15.8 | 3,840 |
| units | unanswerable | 8 | 0 | 0 | 0 | 8 | 8 | 4.75 | 62.0 | 7,576 |
| windows | set1 | 92 | 90 | 2 | 0 | 0 | 0 | 1.03 | 9.2 | 1,827 |
| windows | paraphrase | 177 | 132 | 16 | 1 | 28 | 9 | 1.53 | 15.4 | 2,633 |
| windows | section_id | 50 | 41 | 8 | 1 | 0 | 0 | 1.28 | 11.7 | 2,119 |
| windows | twin_topic | 60 | 49 | 5 | 0 | 6 | 2 | 1.37 | 13.6 | 2,274 |
| windows | multi_section | 90 | 63 | 10 | 5 | 12 | 5 | 1.60 | 17.2 | 2,521 |
| windows | unanswerable | 8 | 0 | 0 | 0 | 8 | 8 | 4.88 | 57.0 | 7,188 |

- On 80% (windows) to 85% (units) of answerable questions its first action is `done`. For those it is an LLM that selects from H1's top 5, not an expansion strategy. That selection is why its Hit@1 is high.
- It uses tools where they are needed: `search` first on 28–33 of the 177 paraphrases and on none of the 92 set 1 questions.
- Its selection is small: about 2 passages and 700 (windows) to 1,000 (units) tokens, which cover 0.92–0.93 of the evidence.
- **Its weak spot is `reference` questions.** Complete evidence is 0.625–0.675, below T0 (0.725–0.825) and far below T3 (0.875–0.900). It finds the citing section, judges it sufficient and stops; `follow_refs` was its first action on 2 to 5 of the 90 multi-section questions.
- The agent's context starts from 5 gated H1 hits. Its evidence in context (0.94–0.95) is therefore bounded by its first stage and by its own searches, and stays below X2 without the gate (0.98–0.99).

**Budget:** 8k is enough. From 4k to 8k evidence in context rises from 0.90–0.93 to 0.94–0.95; from 8k to 16k nothing changes on windows (0.951 to 0.952).

| Method | Index | Gate | Hit@1 | MRR@10 | nDCG@10 (95% CI) | Evidence (context) | Context tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| T0 8k | units | on | 0.678 | 0.784 | 0.754 (0.729–0.778) | 0.939 | 6,281 | 0.02 |
| T2 8k | units | on | 0.678 | 0.779 | 0.741 (0.714–0.767) | 0.907 | 6,963 | 0.02 |
| T3 8k | units | on | 0.678 | 0.747 | 0.760 (0.730–0.789) | 0.909 | 7,886 | 0.02 |
| T2+T3 8k | units | on | 0.678 | 0.746 | 0.752 (0.722–0.782) | 0.883 | 7,905 | 0.02 |
| T4 8k | units | on | 0.928 | 0.946 | 0.867 (0.848–0.885) | 0.942 | 3,977 | 15.49 |
| T0 8k | windows | on | 0.646 | 0.769 | 0.743 (0.718–0.767) | 0.925 | 4,025 | 0.02 |
| T2 8k | windows | on | 0.646 | 0.766 | 0.735 (0.709–0.760) | 0.910 | 6,408 | 0.02 |
| T3 8k | windows | on | 0.646 | 0.735 | 0.762 (0.733–0.791) | 0.912 | 7,428 | 0.02 |
| T2+T3 8k | windows | on | 0.646 | 0.734 | 0.757 (0.727–0.786) | 0.893 | 7,643 | 0.02 |
| T4 8k | windows | on | 0.925 | 0.948 | 0.870 (0.852–0.888) | 0.951 | 2,434 | 14.70 |

Tests at the 8k budget, gate on:

| Comparison | Index | Gate | ΔHit@1 | ΔMRR@10 | ΔnDCG@10 | ΔEvidence (context) |
|---|---|---|---|---|---|---|
| T2 vs T0 | units | on | +0.000 (p=1.000) | **-0.005 (p=0.008)** | **-0.013 (p=0.002)** | **-0.032 (p=0.002)** |
| T2 vs T0 | windows | on | +0.000 (p=1.000) | -0.003 (p=0.193) | **-0.008 (p=0.002)** | -0.015 (p=0.382) |
| T3 vs T0 | units | on | +0.000 (p=1.000) | **-0.037 (p=0.002)** | +0.006 (p=1.000) | **-0.030 (p=0.018)** |
| T3 vs T0 | windows | on | +0.000 (p=1.000) | **-0.034 (p=0.002)** | +0.019 (p=0.081) | -0.013 (p=0.905) |
| T2+T3 vs T0 | units | on | +0.000 (p=1.000) | **-0.038 (p=0.002)** | -0.002 (p=1.000) | **-0.056 (p=0.002)** |
| T2+T3 vs T0 | windows | on | +0.000 (p=1.000) | **-0.034 (p=0.002)** | +0.014 (p=0.495) | -0.032 (p=0.067) |
| T4 vs T2+T3 | units | on | **+0.249 (p=0.002)** | **+0.200 (p=0.002)** | **+0.115 (p=0.002)** | **+0.059 (p=0.002)** |
| T4 vs T2+T3 | windows | on | **+0.279 (p=0.002)** | **+0.214 (p=0.002)** | **+0.113 (p=0.002)** | **+0.058 (p=0.002)** |

By family, ΔEvidence (context), 8k, gate on (exploratory):

| Comparison | Index | Gate | set1 | paraphrase | section_id | twin_topic | multi_section |
|---|---|---|---|---|---|---|---|
| T2 vs T0 | units | on | +0.000 (p=1.000) | -0.040 (p=0.096) | +0.000 (p=1.000) | -0.050 (p=0.987) | **-0.054 (p=0.013)** |
| T2 vs T0 | windows | on | +0.000 (p=1.000) | -0.017 (p=0.450) | -0.007 (p=1.000) | -0.050 (p=1.000) | -0.009 (p=1.000) |
| T3 vs T0 | units | on | -0.011 (p=1.000) | -0.051 (p=0.096) | +0.000 (p=1.000) | -0.050 (p=0.987) | -0.014 (p=1.000) |
| T3 vs T0 | windows | on | -0.011 (p=1.000) | -0.040 (p=0.084) | -0.023 (p=1.000) | -0.017 (p=1.000) | +0.046 (p=0.258) |
| T2+T3 vs T0 | units | on | -0.022 (p=1.000) | **-0.085 (p=0.007)** | +0.000 (p=1.000) | -0.117 (p=0.113) | -0.025 (p=0.977) |
| T2+T3 vs T0 | windows | on | +0.000 (p=1.000) | **-0.085 (p=0.004)** | -0.010 (p=1.000) | -0.050 (p=1.000) | +0.038 (p=0.446) |
| T4 vs T2+T3 | units | on | +0.022 (p=1.000) | **+0.107 (p<0.001)** | +0.020 (p=1.000) | +0.100 (p=0.184) | -0.003 (p=1.000) |
| T4 vs T2+T3 | windows | on | -0.022 (p=1.000) | **+0.147 (p<0.001)** | +0.020 (p=1.000) | +0.050 (p=1.000) | -0.008 (p=1.000) |

The 4k and 16k budgets and the gate-off runs are in the `significance` sheet of each `metrics.xlsx`.

## 10. RQ7: Do 500-token windows beat structure-bounded units?

**Answer:** it depends on the signal, and the plan's guess was wrong. The plan asked whether dense gains more from small windows than BM25 and ColBERT do. Dense does not gain at all.

- **X1 gains clearly:** +0.079 Hit@1, +0.058 MRR, +0.050 nDCG, all with p < 0.001. The bge reranker works much better on 500-token passages than on units of up to 1,800 tokens. On paraphrases its Hit@1 goes from 0.537 to 0.684.
- **ColBERT gains a little:** +0.026 MRR (p = 0.023).
- **BM25 and dense do not change in ranking and lose a little evidence:** Evidence@10 −0.020 and −0.023 (p = 0.041 and 0.040). Ten windows cover less text than ten units.
- **H1 is nominally worse on windows:** Hit@1 −0.034 (p = 0.073), and significantly lower on Evidence@1 and Evidence@5.

Bold marks Holm p < 0.05 for that metric. "Smallest Holm p" is over every section-level, group-level and evidence metric in `granularity.csv`.

| Signal | Hit@1 units → windows | MRR@10 units → windows | nDCG@10 units → windows | Evidence@10 units → windows | Smallest Holm p |
|---|---|---|---|---|---|
| L1 | 0.586 → 0.595 (+0.009) | 0.684 → 0.689 (+0.005) | 0.662 → 0.665 (+0.004) | **0.810 → 0.790 (-0.020)** | 0.003 |
| D1 | 0.659 → 0.667 (+0.009) | 0.765 → 0.777 (+0.012) | 0.749 → 0.755 (+0.007) | **0.912 → 0.889 (-0.023)** | 0.022 |
| C1 | 0.672 → 0.697 (+0.026) | **0.772 → 0.798 (+0.026)** | 0.752 → 0.767 (+0.014) | 0.908 → 0.912 (+0.004) | 0.006 |
| H1 | 0.680 → 0.646 (-0.034) | 0.767 → 0.752 (-0.015) | 0.745 → 0.739 (-0.005) | 0.901 → 0.886 (-0.014) | <0.001 |
| X1 | **0.695 → 0.774 (+0.079)** | **0.788 → 0.846 (+0.058)** | **0.747 → 0.796 (+0.050)** | 0.912 → 0.930 (+0.018) | <0.001 |

Hit@1 by family, units → windows (descriptive):

| Signal | set1 | paraphrase | section_id | twin_topic | multi_section |
|---|---|---|---|---|---|
| L1 | 0.837 → 0.859 | 0.412 → 0.390 | 0.740 → 0.680 | 0.300 → 0.367 | 0.778 → 0.833 |
| D1 | 0.826 → 0.837 | 0.576 → 0.610 | 0.800 → 0.740 | 0.483 → 0.433 | 0.689 → 0.722 |
| C1 | 0.815 → 0.891 | 0.621 → 0.621 | 0.780 → 0.840 | 0.367 → 0.367 | 0.767 → 0.789 |
| H1 | 0.891 → 0.848 | 0.497 → 0.486 | 0.880 → 0.800 | 0.517 → 0.450 | 0.822 → 0.800 |
| X1 | 0.902 → 0.967 | 0.537 → 0.684 | 0.940 → 0.900 | 0.483 → 0.500 | 0.800 → 0.867 |

**The two leaders** (not pre-declared, section 12): X2 does not differ between the indexes on Hit@1 or MRR; units have slightly more evidence in the top 5 (0.982 against 0.963, p = 0.019) and the difference is gone by the full list (0.990 against 0.984). T4 does not differ at all.

**Practical reading:** windows are the better index for a pipeline that ends in a reranker or the agent. Quality is the same, the top 10 costs about 3,800 tokens instead of 6,800, and X2 runs in 8 s instead of 25 s. Units are better only for a pipeline that stops at H1. These windows never cross a section boundary, so nothing here applies to naive fixed-size chunking.

## 11. Score gate (a design variable, not an RQ)

In run 1 the gate was free: after fusion or reranking it never lost a gold section. That is still true for the `set1` family. On the harder families it is not.

| Run | Context tokens, off → on | Latency (s), off → on | Hit@1, off → on | Evidence (context), off → on | Gate loss |
|---|---|---|---|---|---|
| H1 units | 16,052 → 7,353 | 0.02 → 0.02 | 0.680 → 0.678 | 0.947 → 0.948 | 0.019 |
| H1 windows | 7,983 → 4,042 | 0.02 → 0.02 | 0.646 → 0.646 | 0.950 → 0.925 | 0.021 |
| X1 units | 20,136 → 5,183 | 6.99 → 1.41 | 0.695 → 0.708 | 0.952 → 0.891 | 0.041 |
| X1 windows | 8,328 → 2,551 | 2.40 → 0.52 | 0.774 → 0.768 | 0.968 → 0.897 | 0.045 |
| X2 units | 13,760 → 4,698 | 25.15 → 1.58 | 0.838 → 0.844 | 0.990 → 0.945 | 0.030 |
| X2 windows | 7,635 → 2,460 | 8.23 → 0.71 | 0.853 → 0.846 | 0.984 → 0.920 | 0.036 |
| D1 windows | 7,764 → 2,482 | 0.02 → 0.02 | 0.667 → 0.667 | 0.948 → 0.858 | 0.051 |
| L1 windows | 8,217 → 2,460 | 0.02 → 0.02 | 0.595 → 0.595 | 0.839 → 0.761 | 0.049 |

Gate loss by family (share of questions whose gold section the gate removed from the final list):

| Method | Index | all | set1 | paraphrase | section_id | twin_topic | multi_section |
|---|---|---|---|---|---|---|---|
| L1 | units | 0.049 | 0.011 | 0.079 | 0.020 | 0.050 | 0.044 |
| D1 | units | 0.053 | 0.022 | 0.090 | 0.080 | 0.050 | 0.000 |
| H1 | units | 0.019 | 0.000 | 0.040 | 0.000 | 0.033 | 0.000 |
| X1 | units | 0.041 | 0.000 | 0.073 | 0.000 | 0.083 | 0.011 |
| X2 | units | 0.030 | 0.000 | 0.068 | 0.000 | 0.033 | 0.000 |
| L1 | windows | 0.049 | 0.011 | 0.073 | 0.020 | 0.117 | 0.011 |
| D1 | windows | 0.051 | 0.022 | 0.090 | 0.020 | 0.067 | 0.011 |
| H1 | windows | 0.021 | 0.000 | 0.028 | 0.000 | 0.083 | 0.000 |
| X1 | windows | 0.045 | 0.000 | 0.073 | 0.000 | 0.100 | 0.022 |
| X2 | windows | 0.036 | 0.000 | 0.062 | 0.000 | 0.083 | 0.011 |

What happens:

- **Gate A is the cause.** Each first-stage list is cut to a median of 5 hits (the minimum the gate allows). On paraphrases the gold hit often sits below the head of a list, with a weak score. BM25's gate cuts a gold section that was in its top 50 on 11% of questions, dense's on 7–8%.
- **Fusion repairs most of it** (H1 loses 2%), because the other list often still has the hit.
- **A reranker cannot repair it.** It sees about 10 candidates instead of 50. X2's own output gate (gate B) loses almost nothing (0.000 and 0.002); X1's loses 2–4%.
- **The gate does help H1's own ranking.** With the gate on, H1's Hit@3 rises by 0.04 to 0.05 and Evidence@5 by 0.05 (exploratory, raw p < 0.01). RRF over two full top-50 lists rewards passages that are mediocre in both; cutting each list to its strong head first removes them.

**A fixed top-k cut after the reranker is the better way to shorten the context.** Evidence recall and context tokens at each cut-off (exploratory: the token counts were computed for this report from `hits.parquet`):

| Run | top 3 | top 5 | top 10 | top 20 (all returned) |
|---|---|---|---|---|
| X2 windows, gate off | 0.936 / 1,059 | 0.963 / 1,820 | 0.981 / 3,758 | 0.984 / 7,632 |
| X2 units, gate off | 0.953 / 1,840 | 0.982 / 3,276 | 0.987 / 6,765 | 0.990 / 13,759 |
| X2 windows, gate on | 0.902 / 1,063 | 0.916 / 1,831 | 0.920 / 2,459 | 0.920 / 2,459 |
| X2 units, gate on | 0.934 / 1,899 | 0.944 / 3,346 | 0.945 / 4,710 | 0.945 / 4,710 |
| X1 windows, gate off | 0.834 / 1,127 | 0.877 / 1,975 | 0.930 / 4,103 | 0.968 / 8,329 |
| H1 windows, gate on | 0.778 / 1,091 | 0.848 / 1,868 | 0.909 / 3,390 | 0.925 / 4,041 |

Each cell is evidence recall / context tokens. X2 on windows without the gate, cut to its top 5, has more evidence (0.963) in fewer tokens (1,820) than X2 with the gate (0.920 in 2,459).

**What the gate still buys is reranker latency:** 8.2 s to 0.7 s on windows and 25.1 s to 1.6 s on units, for 4 to 6 points of evidence.

The gate thresholds are v1 parameters and were not tuned. They are the first candidate for the single v2 revision the plan allows (section 15).

## 12. Winner and other combinations

This is a decision made after seeing the results, on the same questions. The tests in this section were not pre-declared.

### The finalists

| Method | Index | Gate | Hit@1 (95% CI) | Hit@3 | MRR@10 | Evidence@5 | Evidence in context (95% CI) | Context tokens | Latency (s) | LLM in retrieval |
|---|---|---|---|---|---|---|---|---|---|---|
| X2 | windows | off | 0.853 (0.821–0.885) | 0.970 | 0.912 | 0.963 | 0.984 (0.973–0.994) | 7,635 | 8.23 | no |
| X2 | windows | on | 0.846 (0.814–0.878) | 0.947 | 0.895 | 0.916 | 0.920 (0.897–0.942) | 2,460 | 0.71 | no |
| X2 | units | off | 0.838 (0.804–0.870) | 0.979 | 0.907 | 0.982 | 0.990 (0.982–0.998) | 13,760 | 25.15 | no |
| X2 | units | on | 0.844 (0.810–0.876) | 0.962 | 0.899 | 0.944 | 0.945 (0.925–0.964) | 4,698 | 1.58 | no |
| T4 8k | windows | on | 0.925 (0.902–0.949) | 0.970 | 0.948 | 0.929 | 0.951 (0.933–0.968) | 2,434 | 14.70 | yes |
| T4 8k | units | on | 0.928 (0.904–0.951) | 0.959 | 0.946 | 0.938 | 0.942 (0.923–0.961) | 3,977 | 15.49 | yes |
| T4 16k | units | on | 0.934 (0.910–0.955) | 0.970 | 0.953 | 0.950 | 0.954 (0.937–0.970) | 4,359 | 16.31 | yes |
| T4 4k | windows | on | 0.915 (0.889–0.938) | 0.957 | 0.936 | 0.915 | 0.925 (0.903–0.946) | 2,162 | 13.94 | yes |
| X1 | windows | off | 0.774 (0.736–0.810) | 0.900 | 0.846 | 0.877 | 0.968 (0.952–0.983) | 8,328 | 2.40 | no |
| X1 | windows | on | 0.768 (0.729–0.806) | 0.900 | 0.837 | 0.892 | 0.897 (0.870–0.921) | 2,551 | 0.52 | no |
| F1 | units | on | 0.708 (0.667–0.748) | 0.881 | 0.801 | 0.878 | 0.950 (0.930–0.967) | 7,344 | 0.97 | yes |
| H1 | units | on | 0.678 (0.635–0.721) | 0.876 | 0.784 | 0.876 | 0.948 (0.928–0.966) | 7,353 | 0.02 | no |
| H1 | windows | on | 0.646 (0.601–0.689) | 0.878 | 0.769 | 0.848 | 0.925 (0.903–0.947) | 4,042 | 0.02 | no |

Hit@1 by family:

| Method | Index | Gate | all (469) | set1 (92) | paraphrase (177) | section_id (50) | twin_topic (60) | multi_section (90) |
|---|---|---|---|---|---|---|---|---|
| X2 | windows | off | 0.853 | 0.946 | 0.859 | 0.940 | 0.517 | 0.922 |
| X2 | windows | on | 0.846 | 0.957 | 0.836 | 0.960 | 0.500 | 0.922 |
| X2 | units | off | 0.838 | 0.946 | 0.814 | 0.960 | 0.550 | 0.900 |
| X2 | units | on | 0.844 | 0.946 | 0.825 | 0.960 | 0.550 | 0.911 |
| T4 8k | windows | on | 0.925 | 0.957 | 0.932 | 0.980 | 0.783 | 0.944 |
| T4 8k | units | on | 0.928 | 0.978 | 0.893 | 0.980 | 0.817 | 0.989 |
| T4 16k | units | on | 0.934 | 0.978 | 0.910 | 0.980 | 0.817 | 0.989 |
| T4 4k | windows | on | 0.915 | 0.946 | 0.910 | 0.980 | 0.783 | 0.944 |

Evidence in context by family:

| Method | Index | Gate | all (469) | set1 (92) | paraphrase (177) | section_id (50) | twin_topic (60) | multi_section (90) |
|---|---|---|---|---|---|---|---|---|
| X2 | windows | off | 0.984 | 1.000 | 0.972 | 1.000 | 1.000 | 0.974 |
| X2 | windows | on | 0.920 | 1.000 | 0.898 | 0.980 | 0.917 | 0.849 |
| X2 | units | off | 0.990 | 1.000 | 0.983 | 1.000 | 1.000 | 0.983 |
| X2 | units | on | 0.945 | 1.000 | 0.910 | 1.000 | 0.967 | 0.916 |
| T4 8k | windows | on | 0.951 | 0.978 | 0.960 | 1.000 | 0.917 | 0.901 |
| T4 8k | units | on | 0.942 | 1.000 | 0.921 | 1.000 | 0.950 | 0.888 |
| T4 16k | units | on | 0.954 | 1.000 | 0.938 | 1.000 | 0.983 | 0.894 |
| T4 4k | windows | on | 0.925 | 0.967 | 0.921 | 1.000 | 0.917 | 0.854 |

Paired differences between finalists. Each cell is the difference with two p-values, both Holm-corrected across the 10 rows: `p` treats the 469 questions as independent; `pc` treats each set 1 question and its paraphrases as one unit (292 units). Bold means both are below 0.05.

| Difference (first − second) | ΔHit@1 | ΔMRR@10 | ΔEvidence@5 | ΔEvidence in context |
|---|---|---|---|---|
| T4 8k windows − X2 windows, gate off | **+0.072 (p<0.001; pc<0.001)** | **+0.036 (p=0.002; pc=0.002)** | **-0.033 (p=0.029; pc=0.012)** | **-0.033 (p=0.003; pc<0.001)** |
| T4 8k windows − X2 windows, gate on | **+0.079 (p<0.001; pc<0.001)** | **+0.053 (p<0.001; pc<0.001)** | +0.013 (p=0.736; pc=0.877) | +0.031 (p=0.045; pc=0.131) |
| T4 8k units − X2 units, gate off | **+0.090 (p<0.001; pc<0.001)** | **+0.038 (p=0.001; pc<0.001)** | **-0.044 (p<0.001; pc=0.001)** | **-0.048 (p<0.001; pc<0.001)** |
| T4 8k windows − T4 8k units | -0.002 (p=1.000; pc=1.000) | +0.003 (p=0.918; pc=1.000) | -0.009 (p=0.876; pc=0.445) | +0.009 (p=0.776; pc=1.000) |
| T4 8k windows − T4 4k windows | +0.011 (p=1.000; pc=1.000) | +0.012 (p=0.515; pc=1.000) | +0.014 (p=0.386; pc=0.648) | +0.026 (p=0.022; pc=0.094) |
| T4 16k windows − T4 8k windows | -0.006 (p=1.000; pc=1.000) | -0.005 (p=0.658; pc=1.000) | -0.002 (p=0.876; pc=0.877) | +0.001 (p=1.000; pc=1.000) |
| X2 windows, gate off − X2 windows, gate on | +0.006 (p=1.000; pc=1.000) | +0.017 (p=0.145; pc=0.132) | **+0.046 (p<0.001; pc<0.001)** | **+0.064 (p<0.001; pc<0.001)** |
| X2 windows, gate off − X2 units, gate off | +0.015 (p=1.000; pc=1.000) | +0.005 (p=0.918; pc=1.000) | **-0.019 (p=0.019; pc=0.012)** | -0.006 (p=0.394; pc=0.384) |
| X2 windows, gate off − X1 windows, gate off | **+0.079 (p<0.001; pc=0.013)** | **+0.066 (p<0.001; pc<0.001)** | **+0.086 (p<0.001; pc<0.001)** | +0.016 (p=0.045; pc=0.306) |
| T4 8k windows − H1 windows, gate on | **+0.279 (p<0.001; pc<0.001)** | **+0.180 (p<0.001; pc<0.001)** | **+0.082 (p<0.001; pc<0.001)** | +0.026 (p=0.090; pc=0.384) |

### A. Default pick: jina reranker on windows, no gate, top 10

Run: `X2-win500-gateoff-v1-f352b4b8`, cut to its first 10 hits.

| Stage | Setting |
|---|---|
| Chunking | 500 bge-m3 tokens, at least 50 tokens of overlap, windows never cross a section boundary |
| First stage | BM25 (k1 = 1.2, b = 0.75) and bge-m3 dense, top 50 each, no gate |
| Fusion | RRF, score = Σ 1 / (60 + rank), at most 50 candidates |
| Reranker | `jinaai/jina-reranker-v3`, listwise over the 50 candidates |
| Output | top 10 windows (the run returns 20) |

**Results:** Hit@1 0.853, Hit@3 0.970, MRR@10 0.912. Evidence@5 0.963, Evidence@10 0.981, 0.984 over all 20. About 3,800 tokens for the top 10 (1,800 for the top 5, 7,600 for all 20). 8.2 s per question on the RTX 3090. No LLM; deterministic.

**Why this one:**
- Its evidence coverage is 0.97 to 1.00 in every family over the returned list. Cut to the top 10 it is still 0.97 on paraphrases and 0.96 on multi-section questions.
- Cut to the top 5 it still has more evidence in fewer tokens than the agent's context (0.963 in 1,820 tokens against 0.951 in 2,434).
- It needs no LLM, so it is deterministic and has no prompt to maintain.
- It is significantly better than the same pipeline with the gate (+0.064 evidence), than the same pipeline with bge-reranker-v2-m3 (+0.079 Hit@1), and than its own first stage H1 (+0.207 Hit@1, a pre-declared test).

**Its weaknesses:**
- Twin questions: Hit@1 0.517. The evidence is in the top 5 (0.967), but a section of the other engine type is at rank 1 on 43% of them.
- Hit@1 is 0.072 below the agent.
- 8 s per question. The gated variant (C) takes 0.7 s.
- jina-reranker-v3 is CC BY-NC 4.0: fine for this research project, not for commercial use.

The cut-off of 10 was read off these questions. It is a truncation of the stored list, not a tuned model parameter, but a different question mix could favour 8 or 15.

### B. Best early precision: the Qwen agent on windows, 8k budget

Run: `T4-b8k-win500-gateon-v1-8ace188b`. H1 with the gate on, top 5 as the starting context, then the agent (at most 5 steps, 8k tokens).

**Results:** Hit@1 0.925, Hit@3 0.970, MRR@10 0.948. Evidence in context 0.951 in 2,434 tokens; the agent's own selection is about 2 windows and 700 tokens and covers 0.918 of the evidence. 14.7 s, 1.46 LLM calls and about 4,900 prompt tokens per question.

**Why this one:** it has the highest Hit@1 and MRR of all 110 runs, in every family, and the margin over the best non-LLM run is significant (+0.072 Hit@1, +0.036 MRR). It is the only method that handles twin questions (0.783) and heavy paraphrases (0.933) well, and it hands over the smallest context.

**Its weaknesses:** it retrieves less of the evidence than A (−0.033, significant), mostly on `multi_section` (0.901 against 0.974) and `twin_topic` (0.917 against 1.000). On multi-section questions it often stops after the first section that looks sufficient. It takes 15 s on an ordinary question and about a minute on one the document cannot answer. It depends on one model and one prompt.

Units or windows makes no difference for T4 (no metric differs significantly). Windows use 2,434 context tokens against 3,977.

### C. Fast variant of A: the same pipeline with the gate on

Run: `X2-win500-gateon-v1-8b1df080` (gate A τ = 0.4, δ = 0.3, keep 5–50; gate B τ = 0.2, δ = 0.5, keep 5–20).

**Results:** Hit@1 0.846, MRR 0.895, evidence 0.920, about 6.5 hits and 2,460 tokens, 0.71 s. It keeps A's early precision and loses 6 points of evidence, almost all on paraphrases, twin and multi-section questions.

### D. No reranker and no LLM: H1 on units, gate on

Run: `H1-units-gateon-v1-9f5e5fd2`. Hit@1 0.678, Evidence@10 0.937, 7,353 tokens, 0.02 s. With the engine-type filter in front (F1, one short LLM call, about 1 s) Hit@1 is 0.708 and twin questions go from 0.517 to 0.683.

### Which combinations are deterministic

"Deterministic" here means: with the corpus, the index, the model revisions and the parameters fixed, the output depends only on the question text, and no generative model is called.

| Group | Methods | Notes |
|---|---|---|
| **Deterministic, no LLM** | L1, S1, D1, C1, H1, H2, M1, T1, X1, X2, T0, T2, T3, T2+T3 | Embedding, BM25 weighting, RRF, the gate, both rerankers, the id regex and the fixed expansion rules are all plain functions. Layouts A, C and D are in this group. |
| **Deterministic, but not deployable** | F0 | It reads the question's `engine_type` label. It is an upper bound for F1 only. |
| **LLM in retrieval** | Q1, Q2, F1, T4 | Qwen3.8-27B at temperature 0 with thinking off. Layout B is in this group. |

Notes on the deterministic group:
- Dense search in Qdrant is approximate (HNSW). Notebook 07 checks it against exact search.
- Embeddings and reranker scores agree between CPU and GPU to within about 1e-6 and 4e-5, with identical rankings (plan sections 0 and 5).
- jina-reranker-v3 is listwise: a passage's score depends on the other candidates and their order. The same candidate list always gives the same scores, but changing the first stage or the gate changes them.

Notes on the LLM group:
- Every LLM call is cached under a hash of the full request, so the stored runs replay exactly. Notebook 08 re-ran H1 and T4 (units, 8k) and reproduced both `hits.parquet` files.
- Without the cache a repeat call is not guaranteed to give the same text. Temperature 0 removes sampling, but according to the notes taken when the server was set up, vLLM's batch-invariant mode is not available on this GPU, so the output can change with what else is in the batch.
- The methods differ in how much can vary. F1 returns one of three values under a JSON schema, so its variation is bounded and easy to audit. Q1 and Q2 return free text. T4 makes up to five dependent tool calls, so a small change early can change the whole path.
- The prompts are part of the run identity (their hashes are in `config_hash`). Editing a prompt gives a new run, not a changed one.

### LLM in retrieval: what run 2 says for and against

**For:**
- **Best ranking.** T4 has the highest Hit@1 and MRR in every family: 0.925–0.934 pooled against 0.838–0.853 for the best non-LLM run.
- **It understands the question, not just its words.** T4 is the only method that is as good on an implied engine type as on an explicit one (0.80–0.83 against 0.77–0.80), and it loses the least on heavy paraphrases.
- **Small, selected context.** About 2 passages and 700 to 1,000 tokens cover 0.92–0.93 of the evidence.
- **A signal for questions the document cannot answer** (section 13).
- **Cheap LLM calls pay too.** F1 costs one call of about 140 prompt tokens and 1 s, and removes the wrong-engine errors on twin questions. HyDE adds +0.03 to +0.04 evidence to dense.

**Against:**
- **Latency.** T4 takes 14–16 s per question (1.3–1.5 calls), and 57–62 s on unanswerable questions, where it searches until the step limit. HyDE takes 13 s, keyword expansion 5 s.
- **Not deterministic outside the cache**, and tied to one model and one prompt.
- **GPU sharing.** vLLM and the GPU encoder together need more than the card's 24 GB, so they never run at the same time. LLM methods ran in phase B with the encoder on the CPU; the rerankers ran in phase A. A pipeline that wants a GPU reranker and an LLM step for the same question has to run them in two phases or put one of them on the CPU.
- **Less evidence than the best non-LLM run.** 0.94–0.95 against 0.98–0.99. The agent stops when it judges the context sufficient, and on `reference` questions it leaves the context incomplete about a third of the time.
- **An LLM rewrite can make things worse.** Keyword expansion lowers Hit@1 by 0.07 on units, and HyDE is nominally worse than dense on `section_id` questions.
- **Token cost.** T4 reads 4,900 (windows) to 6,500 (units) prompt tokens per question.

### Avoid or skip

| Method | Reason |
|---|---|
| Q2 keyword expansion | Worse than plain BM25 overall; its gain on paraphrases is smaller than switching to dense |
| S1 learned sparse, H2 | S1 is no better than BM25; adding it to the hybrid does not help |
| M1 | Same result as C1, and no gain over dense |
| T2, T3, T2+T3 fixed expansion | No added evidence; T3 variants lower MRR and nearly double the context |
| T1 as written | Fixes summaries, breaks redirects, net zero |
| X1 on units | No gain over H1; 0.315 Hit@1 on heavy paraphrases |
| The gate in front of a reranker, when evidence coverage matters | Caps evidence at 0.92–0.95; use a top-k cut instead |
| X2 on units, gate off | Same quality as on windows at three times the latency (25 s) and nearly twice the tokens |
| BM25 alone | 0.39–0.41 Hit@1 on paraphrased questions |

### Not tested, so do not assume

The plan did not run a full factorial. These combinations have no results:

- **X2's list as the agent's starting context.** A and B fail on different questions: on windows, A misses Hit@1 on 69 questions and B on 35, and only 24 are shared. This is the most promising untested combination.
- **F1 in front of X2.** F1 fixes exactly the family where X2 is weakest (`twin_topic`), but on H1's ranking only.
- HyDE or dense alone as the candidate generator for a reranker.
- A reranker over fewer candidates without the gate (for example the fused top 20), which might keep A's evidence at lower latency.
- Any reranker over ColBERT, and H2 plus a reranker.

## 13. Questions the document cannot answer

Retrieval always returns at least 5 hits, so it does not abstain by itself. Run 1 suggested two signals. Run 2 still has only the 8 unanswerable questions of set 1 (set 2 added none), but it has 377 new answerable questions to test the signals against.

**Reranker top score: no longer usable.** Under X1 all 8 unanswerable questions have a top score of 1.41 or lower. In run 1 only 5 of 92 answerable questions were that low. In run 2, 175 (units) to 184 (windows) of the 469 answerable questions are, including 136–137 of the 177 paraphrases. A paraphrased question that the document does answer looks to the reranker like a question it does not answer. X2's scores separate somewhat better (AUC 0.84–0.85 against 0.74–0.75 for X1) but still leave about 180 answerable questions below the highest unanswerable one.

**Agent behaviour: weaker than in run 1, still useful.** At the 8k budget:

| Signal | Unanswerable (of 8) | Answerable (of 469) |
|---|---|---|
| First action is `search` | 8 | 46 (windows), 48 (units) |
| Three or more searches | 8 | 16 (windows), 11 (units) |

"First action is `search`" was a clean signal in run 1 (0 of 92). It is not any more, because the agent also searches first on paraphrases. "Three or more searches" keeps all 8 unanswerable questions and flags 2–3% of the answerable ones. It rests on 8 questions and costs about a minute per flagged question, so it needs more unanswerable questions before it can be relied on.

## 14. Limitations to share with your advisor

1. **All 477 questions are LLM-generated** (`claude-opus-5-5`). Set 1 is unreviewed (`review.status = unreviewed` on all 100). Set 2 passed the automatic checks and the judge checks of its specs (answer from gold, single source, same meaning, each section alone fails); its records carry `review.status = ok` with no reviewer named, and the optional second judge from another model family is `pending` on all 377.
2. **The families are stress tests, not a sample of user questions.** The pooled numbers depend on the mix: 38% paraphrases, 19% multi-section, 13% twin, 11% section-id, 20% set 1. A different mix gives different pooled numbers and could change close rankings. Quote families separately.
3. **Questions are not independent.** A paraphrase shares its gold with its original and with the other paraphrase, so the 469 questions fall into 292 groups once each paraphrase is counted with its original. Pooled confidence intervals and pre-declared p-values are too narrow. I re-tested the finalist comparisons with each original and its paraphrases as one unit (section 12); the bolded differences hold. The pre-declared tests in sections 4 to 10 were not corrected this way.
4. **Per-family tests, sub-family splits and the finalist tests are exploratory.** Sub-groups such as the 10 redirect or 10 summary questions are too small for more than a description.
5. **Picks are made on the test data.** Section 12 was chosen after seeing these questions, and the top-10 cut-off was read off them. No held-out set exists.
6. **Only v1 parameters were run.** Several findings are about v1 parameter values and not about the methods: the gate's thresholds and minimum of 5, the agent's 5 starting hits, the 50 reranker candidates.
7. **Only 8 unanswerable questions.** Section 13 cannot support a threshold.
8. **One document.** CS-E has a rare structure (CS and AMC pairs, piston and turbine twins). The twin and filter results in particular may not carry over.
9. **No full factorial.** The two leaders were never combined (section 12).
10. **Latency reflects one RTX 3090.** Phase A used the GPU encoder; phase B used Qwen on the GPU and the encoder on the CPU. X1 gate-on latencies are pro-rated from the gate-off batch. LLM latencies are those of the first, uncached call.

## 15. Suggested next steps

1. **Decide the single v2 revision the plan allows.** Run 2's v1 metrics have now been read. The candidates they point to are the gate (a higher minimum for gate A, or no gate A in front of a reranker), the agent's number of starting hits and the reranker's candidate count. The plan allows one hand revision, and v2 would be chosen on these same questions, so it has to be reported as the revised result next to v1.
2. **Test the two untested combinations** from section 12: X2's list as the agent's starting context, and F1 in front of X2. Both are new method configurations, not parameter changes, so they need a line in `plan.md` first. Reranker scores and LLM outputs are cached, so both fit the existing two-phase run.
3. **Add unanswerable questions** in the style of each family (a paraphrased near-miss, a wrong-engine-type question, a non-existent section id). Without them section 13 stays anecdotal.
4. **Review the questions.** A person should review set 1, and the second-judge check of set 2 should be run with Qwen in phase B.
5. **Keep the regression check.** After any change to `evaluate.py` or `report.py`, the `set1` columns must still reproduce `run1resultarchive/`.

## 16. Where the numbers come from

| What | File |
|---|---|
| All run metrics, one row per run, with per-family nDCG columns | `results/run2/units/metrics.csv`, `results/run2/win500/metrics.csv` |
| Pre-declared tests, breakdowns (`by_family`, `by_question_type`, `by_engine_type`, `by_kind`, `mentions_section_id`, `length_bucket`), gate, per-question ranks, unanswerable | sheets of `results/run2/units/metrics.xlsx` and `results/run2/win500/metrics.xlsx` |
| Every metric per family, with n and bootstrap CI | `results/run2/run_metrics.parquet` (`subset = family=<name>`) |
| Windows vs units tests | `results/run2/granularity.csv` |
| Per-question metrics, for new paired tests | `results/run2/question_metrics.parquet` |
| Each run's hits, stats, config and agent trace | `results/run2/runs/<run_id>/` |
| The questions and their family fields | `data/eval/questions.jsonl`, `data/eval/set2/*.jsonl`, specs in `data/eval/specs/` |
| Run 1 values for comparison | `run1resultarchive/` (identical to the `set1` family here) |

Not produced by notebook 09, computed once from the files above for this report: the per-family tests of the pre-declared comparisons; the paraphrase, section-id, twin and hop-type splits; the multi-section table on the 90 `multi_section` questions (the `multi_*` columns in `metrics.csv` have n = 100, because they also include the 10 summary questions); the wrong-engine rate; gate loss by family; evidence and tokens at top-k cut-offs; the finalist comparisons with both p-values; the agent's first actions, search counts and selection size; the F1 extraction check; and the top-score separation.
