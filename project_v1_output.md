# CS-E retrieval experiments: v1 results

**Scope:** 110 runs on EASA CS-E Amendment 9, with parameter set `v1` only. All metrics use the 92 answerable questions; the 8 unanswerable ones are excluded from scoring. Numbers come from `results/units/metrics.*`, `results/win500/metrics.*` and `results/granularity.csv`, which notebook 09 wrote.

**How this document relates to the plan:** `plan.md` says no run is selected as "the winner". The RQ sections below keep to that. They report each pre-declared test and name the leading option as a description only. Section 10, "Combinations to use in real projects", is a separate engineering decision made after seeing the results. It should not be presented as a pre-declared finding.

## 0. Terms

### Single-signal baselines (RQ1)
| id | method | Qdrant mechanism |
|---|---|---|
| L1 | BM25 | `query(bm25 sparse)` |
| S1 | bge-m3 learned sparse | `query(bge_sparse)` |
| D1 | bge-m3 dense | `query(dense)` |
| C1 | ColBERT MaxSim over all units | exact MaxSim over every unit's full ColBERT vectors, client-side (Qdrant's multivector cap, section 3.5) |

### Hybrid and multi-stage (RQ2)
| id | method |
|---|---|
| H1 | BM25 + dense: each top-50 → gate A → **RRF** |
| H2 | BM25 + bge-sparse + dense: each top-50 → gate A → RRF (three-way) |
| M1 | dense top-50 → gate A → **ColBERT rerank** of the kept hits (exact MaxSim over their full ColBERT vectors, client-side; section 3.5) → gate B |

### Metadata and structure (RQ5), on top of H1
| id | method |
|---|---|
| F0 | **oracle filter**: the same filter as F1, but built from the question's `engine_type` label instead of the LLM. `piston` / `turbine` → `engine_type ∈ {label, all}`; `all` → no filter. It is the upper bound for F1, and the only method that reads a label. |
| F1 | **LLM-extracted filter**: Qwen returns JSON (schema `{engine_type: "piston" \| "turbine" \| null}`), and the filter becomes `engine_type ∈ {extracted, all}`; `null` means no filter. The filter is applied inside each signal's Qdrant search, before gate A. |
| T1 | **id lookup**: a regex finds "CS-E 740", "AMC E 20(f)" in the question text; that section's units or windows go first, then H1's results |

### Query-side LLM methods (RQ4; cached in `data/cache/llm.jsonl`)
| id | method |
|---|---|
| Q1 | **HyDE**: Qwen writes one hypothetical CS/AMC-style passage that answers the question. The dense query vector is the normalised mean of the question's and the passage's embeddings. Dense search → gate A; compared with D1. |
| Q2 | **keyword expansion for BM25**: Qwen lists regulation-style terms for the question (e.g. "flame out" → "flame-out, relight"). BM25 runs on the question plus these terms, so no original word is lost → gate A; compared with L1. |

### Rerankers (RQ3), on H1's fused list
| id | method |
|---|---|
| X1 | **bge-reranker-v2-m3**: pointwise cross-encoder, `encoder-gpu` (phase A) → gate B |
| X2 | **jina-reranker-v3**: listwise (the query and all candidates in one context), `encoder-gpu` (phase A) → gate B |

### Expansion: fixed rules and agent (RQ6), on H1
| id | method |
|---|---|
| T0 | **no expansion**: H1's units or windows in rank order, overlapping windows merged, cut to the token budget (the reference point for every expansion method) |
| T2 | **fixed parent expansion** (small-to-big, baseline): H1's hits in rank order → their sections (or groups), deduplicated, until the token budget is full. For windows this is the classic small-to-big. |
| T3 | **fixed cross-reference following** (baseline): H1's hits in rank order, each followed by the sections it references (in order of mention), until the token budget is full. It always includes the AMCs of a retrieved CS. |
| T2+T3 | both fixed rules together, under the same token budget |
| T4 | **agentic small-to-big** (single question, step limit, token budget). It starts from H1's hits, not from an empty context. Qwen gets the tools `expand(unit_id, to="section"\|"group")`, `follow_refs(unit_id)`, `search(query)` and `done(evidence_ids)`. It decides per question whether the hits are already enough (it calls `done` straight away), or whether to expand, follow a reference or search again. `search` uses H1 with the run's gate. Every step is logged with its decision, tool and arguments, and the tokens added. |

### Everything in a single list
| L1 | BM25 |
| S1 | bge-m3 learned sparse |
| D1 | bge-m3 dense |
| C1 | ColBERT MaxSim over all units |
| H1 | BM25 + dense: each top-50 → gate A → **RRF** |
| H2 | BM25 + bge-sparse + dense: each top-50 → gate A → RRF (three-way) |
| M1 | dense top-50 → gate A → **ColBERT rerank** of the kept hits  → gate B |
| F0 | **oracle filter**: same filter as F1, built from the question's `engine_type` instead of LLM. |
| F1 | **LLM-extracted filter**: Qwen returns JSON|
| T1 | **id lookup**: a regex finds "CS-E 740" in qst text, units or windows go first, then H1's results |
| Q1 | **HyDE**: Qwen writes one hypothetical passage |
| Q2 | **keyword expansion for BM25**: Qwen lists regulation-style terms |
| X1 | **bge-reranker-v2-m3**: pointwise cross-encoder,|
| X2 | **jina-reranker-v3**: listwise |
| T0 | **no expansion**: H1's units or windows in rank order |
| T2 | **fixed parent expansion** (small-to-big, baseline) |
| T3 | **fixed cross-reference following** (baseline):  |
| T2+T3 | both fixed rules together |
| T4 | **agentic small-to-big** (single question, step limit, token budget). |

## 1. Summary

| RQ | Question | Short answer | Leading option | Pre-declared tests (Holm-corrected) |
|---|---|---|---|---|
| RQ1 | Best single signal? | No signal clearly beats BM25 | ColBERT (C1), on windows only | None significant |
| RQ2 | Does hybrid beat its parts? | Small gain that is not significant, but it closes the recall gaps | BM25 + dense RRF (H1) | None significant |
| RQ3 | Do rerankers help, and which one? | **Yes on windows.** On units only X2's nDCG improves. The two rerankers are close | X1 and X2 are tied | X1 and X2 beat H1 on windows, 11 of 12 tests |
| RQ4 | Do LLM query rewrites help? | **No.** HyDE adds nothing and keyword expansion hurts | Plain queries | Q2 significantly worse than BM25; Q1 no difference |
| RQ5 | Do metadata and structure help? | Barely, on this question set | LLM filter (F1), marginally | F1 nDCG on windows, gate off, only. T1 has no effect |
| RQ6 | Does expansion help? Is the agent better? | Fixed expansion does not add evidence here. The agent improves early precision on windows | Agent (T4) | T4 beats T2+T3 on MRR on windows. T3 significantly lowers MRR in several runs |
| RQ7 | Windows or units? | No significant difference. Windows use about half the context | Windows, for context cost | None significant |

**Combination to use (section 10):** 500-token windows, BM25 + bge-m3 dense fused with RRF, score gate on, then `bge-reranker-v2-m3`. On this set the correct section is ranked first on 96.7% of questions and is always in the top 3. It hands over about 2,600 tokens per question and takes about 0.5 s on the RTX 3090.

**Biggest caveat:** all 100 questions were generated by Claude and none has been reviewed by a person (section 11). Treat the numbers as provisional until the questions are reviewed.

## 2. How to read "best"

No single number decides this. I ranked options by asking these questions in order:

1. **Is the evidence retrieved at all?** Evidence recall: the gold quote's text is covered by the retrieved chunks. If this is low, the answer cannot be generated. It compares fairly across both indexes.
2. **Is the right section near the top?** Hit@1 and MRR@10 at section level. They count only the true gold section, so they are the cleanest early-precision measures here.
3. **What does it cost?** Context tokens handed to the LLM and latency.
4. **Is a difference real?** A difference only counts as a finding if its Holm-corrected p-value is below 0.05 on the pre-declared comparison. With 92 questions, differences of 0.01–0.03 are usually noise. Most confidence intervals overlap.

**Why not just sort by nDCG@10 (the table's default sort key).** Section nDCG gives partial credit (grade 1) to sibling sections in the gold section's group, typically the CS and AMC pair. Methods that add or keep siblings gain nDCG without finding the gold section any better:

- T3 (cross-reference following, which always adds the AMC of a retrieved CS) has 0.73 grade-1 siblings in its top 10 against T0's 0.48 (units, 8k, gate on). The gold section count stays at about 1.0. Its nDCG rises by 0.035 while its MRR falls by 0.021.
- X2 on units, gate off, tops the nDCG table at 0.922. Turning the gate on drops nDCG to 0.900, but Hit@1 (0.946) and MRR (0.971) do not change. The only thing lost is siblings: 0.64 → 0.47 in the top 10.

So X2 gate-off at the top of the table does not mean it is the best method. It costs 27 s and 13,775 tokens per question for that nDCG.

**Method codes:** L1 BM25 · S1 bge-m3 learned sparse · D1 bge-m3 dense · C1 ColBERT · H1 BM25 + dense RRF · H2 BM25 + sparse + dense RRF · M1 dense then ColBERT rerank · X1 bge-reranker-v2-m3 · X2 jina-reranker-v3 · Q1 HyDE · Q2 keyword expansion · F0 oracle engine-type filter · F1 LLM engine-type filter · T1 section-id lookup · T0 no expansion · T2 parent section · T3 cross-references · T4 agent.

**Table conventions:** Hit@1, MRR and nDCG are at section level. "Units" are the 252 structure-bounded base units, "windows" the 436 section-bounded 500-token windows. Δ is the first method minus the second. **Bold** means Holm-corrected p < 0.05. Latency is per question on the RTX 3090.

## 3. RQ1: Which single retrieval signal works best?

**Answer:** none is significantly better than BM25. The tests compared S1, D1 and C1 against L1 and found nothing at either gate setting on either index. ColBERT is nominally ahead on windows (Hit@1 0.891 vs BM25's 0.859). On units BM25 is ahead on Hit@1 (0.837 vs 0.815). Learned sparse is nominally last on both.

**Practical reading:** BM25 is a strong baseline for this text and costs almost nothing. ColBERT takes 1.4–1.9 s per query and needs a client-side index, for no proven gain.

| Method | Index | Gate | Hit@1 | MRR@10 | nDCG@10 (95% CI) | Evidence@10 | Context tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| L1 | units | off | 0.837 | 0.894 | 0.846 (0.800–0.888) | 0.978 | 16,714 | 0.04 |
| S1 | units | off | 0.761 | 0.855 | 0.823 (0.779–0.865) | 0.989 | 17,720 | 0.03 |
| D1 | units | off | 0.826 | 0.882 | 0.848 (0.802–0.891) | 0.967 | 14,203 | 0.03 |
| C1 | units | off | 0.815 | 0.892 | 0.863 (0.823–0.900) | 0.989 | 11,941 | 1.90 |
| L1 | windows | off | 0.859 | 0.905 | 0.852 (0.808–0.893) | 0.978 | 8,082 | 0.04 |
| S1 | windows | off | 0.804 | 0.881 | 0.845 (0.804–0.884) | 0.978 | 8,154 | 0.03 |
| D1 | windows | off | 0.837 | 0.896 | 0.856 (0.814–0.896) | 0.978 | 7,756 | 0.03 |
| C1 | windows | off | 0.891 | 0.935 | 0.880 (0.845–0.913) | 1.000 | 7,406 | 1.44 |

| Comparison | Index | Gate | ΔHit@1 | ΔMRR@10 | ΔnDCG@10 | ΔEvidence@10 |
|---|---|---|---|---|---|---|
| S1 vs L1 | units | off | -0.076 (p=1.000) | -0.039 (p=1.000) | -0.023 (p=1.000) | +0.011 (p=1.000) |
| S1 vs L1 | windows | off | -0.054 (p=1.000) | -0.024 (p=1.000) | -0.007 (p=1.000) | +0.000 (p=1.000) |
| D1 vs L1 | units | off | -0.011 (p=1.000) | -0.012 (p=1.000) | +0.002 (p=1.000) | -0.011 (p=1.000) |
| D1 vs L1 | windows | off | -0.022 (p=1.000) | -0.009 (p=1.000) | +0.004 (p=1.000) | +0.000 (p=1.000) |
| C1 vs L1 | units | off | -0.022 (p=1.000) | -0.002 (p=1.000) | +0.017 (p=1.000) | +0.011 (p=1.000) |
| C1 vs L1 | windows | off | +0.033 (p=1.000) | +0.030 (p=1.000) | +0.028 (p=0.636) | +0.022 (p=1.000) |

Gate-on results lead to the same conclusion: no comparison is significant.

## 4. RQ2: Does hybrid fusion beat the single signals?

**Answer:** H1 is nominally better than both BM25 and dense, but no pre-declared comparison is significant. The closest are H1 vs L1 and H1 vs D1 on units with the gate on (nDCG p = 0.067 for both). H2 (adding learned sparse) and M1 (ColBERT reranking of dense) do not significantly improve on H1 and D1.

**Practical reading:** H1 is the best first stage even without a significant win, for a recall reason. With the gate on, its evidence recall@10 is 1.000 on both indexes, against 0.967 and 0.957 for dense, and 0.967 for BM25. BM25 and dense miss different questions, and fusion covers both. For a reranking pipeline, what matters is getting the evidence into the candidate list.

| Method | Index | Gate | Hit@1 | MRR@10 | nDCG@10 (95% CI) | Evidence@10 | Context tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| L1 | units | off | 0.837 | 0.894 | 0.846 (0.800–0.888) | 0.978 | 16,714 | 0.04 |
| D1 | units | off | 0.826 | 0.882 | 0.848 (0.802–0.891) | 0.967 | 14,203 | 0.03 |
| H1 | units | off | 0.891 | 0.928 | 0.870 (0.830–0.908) | 0.989 | 15,977 | 0.04 |
| H2 | units | off | 0.848 | 0.907 | 0.868 (0.830–0.904) | 1.000 | 16,795 | 0.04 |
| M1 | units | off | 0.815 | 0.892 | 0.863 (0.823–0.900) | 0.989 | 12,276 | 0.17 |
| L1 | windows | off | 0.859 | 0.905 | 0.852 (0.808–0.893) | 0.978 | 8,082 | 0.04 |
| D1 | windows | off | 0.837 | 0.896 | 0.856 (0.814–0.896) | 0.978 | 7,756 | 0.03 |
| H1 | windows | off | 0.848 | 0.911 | 0.866 (0.829–0.900) | 1.000 | 7,871 | 0.04 |
| H2 | windows | off | 0.859 | 0.917 | 0.878 (0.842–0.910) | 0.989 | 7,954 | 0.04 |
| M1 | windows | off | 0.891 | 0.935 | 0.881 (0.846–0.914) | 1.000 | 7,460 | 0.40 |

| Comparison | Index | Gate | ΔHit@1 | ΔMRR@10 | ΔnDCG@10 | ΔEvidence@10 |
|---|---|---|---|---|---|---|
| H1 vs L1 | units | off | +0.054 (p=1.000) | +0.034 (p=0.526) | +0.024 (p=0.811) | +0.011 (p=1.000) |
| H1 vs L1 | windows | off | -0.011 (p=1.000) | +0.006 (p=1.000) | +0.014 (p=0.842) | +0.022 (p=1.000) |
| H1 vs D1 | units | off | +0.065 (p=1.000) | +0.046 (p=0.449) | +0.022 (p=0.943) | +0.022 (p=1.000) |
| H1 vs D1 | windows | off | +0.011 (p=1.000) | +0.015 (p=1.000) | +0.009 (p=0.842) | +0.022 (p=1.000) |
| H2 vs H1 | units | off | -0.043 (p=1.000) | -0.022 (p=0.526) | -0.003 (p=1.000) | +0.011 (p=1.000) |
| H2 vs H1 | windows | off | +0.011 (p=1.000) | +0.005 (p=1.000) | +0.012 (p=0.530) | -0.011 (p=1.000) |
| M1 vs D1 | units | off | -0.011 (p=1.000) | +0.010 (p=1.000) | +0.015 (p=0.943) | +0.022 (p=1.000) |
| M1 vs D1 | windows | off | +0.054 (p=1.000) | +0.038 (p=0.631) | +0.025 (p=0.530) | +0.022 (p=1.000) |

Gate-on tests are also all non-significant.

## 5. RQ3: Does a reranker improve early precision, and how do the two compare?

**Answer:** yes on windows. X1 beats H1 significantly on Hit@1, MRR and nDCG at both gate settings. X2 does too, except Hit@1 with the gate off (p = 0.067). On units the reranker effect is weaker: only X2's nDCG with the gate off is significant, and part of that is the sibling effect from section 2.

**X1 vs X2:** no significant difference on Hit@1 or MRR anywhere. X2's nDCG is significantly higher only on units with the gate off. On windows they are effectively tied (gate on: X1 Hit@1 0.967, X2 0.957).

**Practical reading:** adding a reranker is the single most useful change in this study. X1 has a licensing advantage for commercial use (section 10). Latency also differs: X2 is listwise and slow on long inputs with the gate off (27 s on units).

| Method | Index | Gate | Hit@1 | MRR@10 | nDCG@10 (95% CI) | Evidence@10 | Context tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| H1 | units | off | 0.891 | 0.928 | 0.870 (0.830–0.908) | 0.989 | 15,977 | 0.04 |
| H1 | units | on | 0.902 | 0.943 | 0.876 (0.840–0.911) | 1.000 | 6,276 | 0.04 |
| X1 | units | off | 0.902 | 0.951 | 0.885 (0.855–0.914) | 1.000 | 19,756 | 7.40 |
| X1 | units | on | 0.902 | 0.951 | 0.887 (0.857–0.916) | 1.000 | 5,207 | 1.34 |
| X2 | units | off | 0.946 | 0.971 | 0.922 (0.896–0.947) | 1.000 | 13,775 | 27.07 |
| X2 | units | on | 0.946 | 0.971 | 0.900 (0.871–0.928) | 1.000 | 5,024 | 1.35 |
| H1 | windows | off | 0.848 | 0.911 | 0.866 (0.829–0.900) | 1.000 | 7,871 | 0.04 |
| H1 | windows | on | 0.837 | 0.910 | 0.865 (0.830–0.899) | 1.000 | 3,430 | 0.04 |
| X1 | windows | off | 0.967 | 0.984 | 0.903 (0.876–0.930) | 1.000 | 8,270 | 2.52 |
| X1 | windows | on | 0.967 | 0.984 | 0.906 (0.877–0.933) | 1.000 | 2,619 | 0.49 |
| X2 | windows | off | 0.946 | 0.973 | 0.921 (0.897–0.945) | 1.000 | 7,555 | 8.86 |
| X2 | windows | on | 0.957 | 0.978 | 0.907 (0.878–0.934) | 1.000 | 2,592 | 0.62 |

| Comparison | Index | Gate | ΔHit@1 | ΔMRR@10 | ΔnDCG@10 | ΔEvidence@10 |
|---|---|---|---|---|---|---|
| X1 vs H1 | units | off | +0.011 (p=1.000) | +0.023 (p=1.000) | +0.015 (p=0.690) | +0.011 (p=1.000) |
| X1 vs H1 | units | on | +0.000 (p=1.000) | +0.008 (p=1.000) | +0.011 (p=0.690) | +0.000 (p=1.000) |
| X1 vs H1 | windows | off | **+0.120 (p=0.006)** | **+0.072 (p=0.004)** | **+0.037 (p=0.013)** | +0.000 (p=1.000) |
| X1 vs H1 | windows | on | **+0.130 (p=0.004)** | **+0.073 (p=0.004)** | **+0.041 (p=0.003)** | +0.000 (p=1.000) |
| X2 vs H1 | units | off | +0.054 (p=1.000) | +0.043 (p=0.433) | **+0.052 (p=<0.001)** | +0.011 (p=1.000) |
| X2 vs H1 | units | on | +0.043 (p=1.000) | +0.028 (p=1.000) | +0.024 (p=0.258) | +0.000 (p=1.000) |
| X2 vs H1 | windows | off | +0.098 (p=0.067) | **+0.062 (p=0.026)** | **+0.056 (p=0.001)** | +0.000 (p=1.000) |
| X2 vs H1 | windows | on | **+0.120 (p=0.014)** | **+0.068 (p=0.012)** | **+0.042 (p=0.002)** | +0.000 (p=1.000) |
| X1 vs X2 | units | off | -0.043 (p=1.000) | -0.020 (p=1.000) | **-0.037 (p=<0.001)** | +0.000 (p=1.000) |
| X1 vs X2 | units | on | -0.043 (p=1.000) | -0.020 (p=1.000) | -0.013 (p=0.258) | +0.000 (p=1.000) |
| X1 vs X2 | windows | off | +0.022 (p=1.000) | +0.011 (p=1.000) | -0.018 (p=0.074) | +0.000 (p=1.000) |
| X1 vs X2 | windows | on | +0.011 (p=1.000) | +0.005 (p=1.000) | -0.001 (p=0.848) | +0.000 (p=1.000) |

Latency note: X1's gate-on figures are a pro-rata share of the gate-off batch, because its scores are cached per (query, unit). X2's gate-on figures are measured directly.

## 6. RQ4: Do LLM query rewrites help?

**Answer:** no. HyDE (Q1) is not significantly better than dense, and costs about 13 s per query on the Qwen setup. Keyword expansion (Q2) is **significantly worse** than plain BM25: Hit@1 falls by 0.13–0.15.

**When the wording differs from the regulation (exploratory, not in notebook 09):** I split questions into three groups by how many of their content words appear in the gold section. HyDE shows no benefit in any group. Keyword expansion is worst on the low-overlap group, where it was supposed to help most.

| MRR@10 difference (rewrite minus plain, gate off) | Low overlap (36 questions) | Medium (27) | High (29) |
|---|---|---|---|
| Q1 − D1, units | −0.009 | +0.016 | +0.026 |
| Q1 − D1, windows | +0.009 | −0.016 | +0.003 |
| Q2 − L1, units | −0.183 (raw p = 0.002) | −0.124 (raw p = 0.049) | −0.046 |
| Q2 − L1, windows | −0.147 (raw p = 0.003) | −0.101 (raw p = 0.065) | −0.026 |

These p-values are uncorrected and this split was not pre-declared. The set also has little real wording gap: on average 80% of a question's content words appear in its gold section. That limits what RQ4 can show (section 11).

| Method | Index | Gate | Hit@1 | MRR@10 | nDCG@10 (95% CI) | Evidence@10 | Context tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| L1 | units | off | 0.837 | 0.894 | 0.846 (0.800–0.888) | 0.978 | 16,714 | 0.04 |
| Q2 | units | off | 0.685 | 0.772 | 0.753 (0.692–0.811) | 0.935 | 18,157 | 4.84 |
| D1 | units | off | 0.826 | 0.882 | 0.848 (0.802–0.891) | 0.967 | 14,203 | 0.03 |
| Q1 | units | off | 0.826 | 0.891 | 0.860 (0.817–0.899) | 0.978 | 13,065 | 13.08 |
| L1 | windows | off | 0.859 | 0.905 | 0.852 (0.808–0.893) | 0.978 | 8,082 | 0.04 |
| Q2 | windows | off | 0.728 | 0.810 | 0.780 (0.721–0.835) | 0.924 | 8,349 | 4.84 |
| D1 | windows | off | 0.837 | 0.896 | 0.856 (0.814–0.896) | 0.978 | 7,756 | 0.03 |
| Q1 | windows | off | 0.826 | 0.896 | 0.866 (0.826–0.904) | 0.989 | 7,603 | 13.08 |

| Comparison | Index | Gate | ΔHit@1 | ΔMRR@10 | ΔnDCG@10 | ΔEvidence@10 |
|---|---|---|---|---|---|---|
| Q1 vs D1 | units | off | +0.000 (p=1.000) | +0.009 (p=1.000) | +0.011 (p=0.403) | +0.011 (p=1.000) |
| Q1 vs D1 | windows | off | -0.011 (p=1.000) | -0.000 (p=1.000) | +0.009 (p=0.643) | +0.011 (p=1.000) |
| Q2 vs L1 | units | off | **-0.152 (p=0.007)** | **-0.123 (p=<0.001)** | **-0.093 (p=<0.001)** | -0.043 (p=0.368) |
| Q2 vs L1 | windows | off | **-0.130 (p=0.007)** | **-0.095 (p=<0.001)** | **-0.072 (p=<0.001)** | -0.054 (p=0.181) |

The gate-on results lead to the same conclusion.

## 7. RQ5: Do metadata and document structure help?

**Answer:** only slightly. The LLM engine-type filter (F1) significantly improves nDCG in one case, on windows with the gate off (+0.019, p = 0.001). Hit@1 and MRR do not significantly change. Section-id lookup (T1) changes nothing at all, for a specific reason: only 6 of the 92 questions mention a section id, and those already have MRR 1.000 without the lookup. So T1's value was not measurable here.

The LLM filter matches or slightly exceeds the oracle (F0), so extraction quality is not what limits it. F0 is reported as a bound and is not tested.

**Practical reading:** F1 adds about 1 s per question and needs an engine-type field, which is specific to this corpus. T1 costs almost nothing and may help if real users type section ids, but this study cannot confirm that.

| Method | Index | Gate | Hit@1 | MRR@10 | nDCG@10 (95% CI) | Evidence@10 | Context tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| H1 | units | off | 0.891 | 0.928 | 0.870 (0.830–0.908) | 0.989 | 15,977 | 0.04 |
| F0 | units | off | 0.891 | 0.929 | 0.877 (0.838–0.913) | 0.989 | 15,999 | 0.04 |
| F1 | units | off | 0.891 | 0.929 | 0.878 (0.839–0.914) | 0.989 | 16,012 | 0.99 |
| T1 | units | off | 0.891 | 0.928 | 0.870 (0.830–0.908) | 0.989 | 15,977 | 0.04 |
| H1 | windows | off | 0.848 | 0.911 | 0.866 (0.829–0.900) | 1.000 | 7,871 | 0.04 |
| F0 | windows | off | 0.859 | 0.917 | 0.881 (0.845–0.914) | 1.000 | 7,874 | 0.04 |
| F1 | windows | off | 0.870 | 0.922 | 0.885 (0.849–0.917) | 1.000 | 7,868 | 0.99 |
| T1 | windows | off | 0.848 | 0.911 | 0.866 (0.829–0.900) | 1.000 | 7,871 | 0.04 |

| Comparison | Index | Gate | ΔHit@1 | ΔMRR@10 | ΔnDCG@10 | ΔEvidence@10 |
|---|---|---|---|---|---|---|
| F1 vs H1 | units | off | +0.000 (p=1.000) | +0.000 (p=1.000) | +0.008 (p=0.909) | +0.000 (p=1.000) |
| F1 vs H1 | windows | off | +0.022 (p=1.000) | +0.011 (p=1.000) | **+0.019 (p=<0.001)** | +0.000 (p=1.000) |
| T1 vs H1 | units | off | +0.000 (p=1.000) | +0.000 (p=1.000) | +0.000 (p=1.000) | +0.000 (p=1.000) |
| T1 vs H1 | windows | off | +0.000 (p=1.000) | +0.000 (p=1.000) | +0.000 (p=1.000) | +0.000 (p=1.000) |

## 8. RQ6: Does expansion bring more evidence? Does an agent beat fixed rules?

**Fixed rules:** no. T2, T3 and T2+T3 never significantly increase evidence recall over T0, and T0 is already at 1.000 with the gate on at 8k. T3 and T2+T3 **significantly lower MRR** in several runs, all at the 8k and 16k budgets, because added sections push hits down the list. Their higher nDCG is the sibling effect from section 2.

**Agent vs fixed rules:** the agent improves early precision over T2+T3 on windows: MRR is significantly higher at all three budgets (+0.079 to +0.088), and Hit@1 at 8k and 16k. On units the MRR gain is similar (+0.063 to +0.069) but not significant (p = 0.070–0.076).

**What the agent actually does.** On answerable questions it calls `done` on its first step in 90–92 of 92 cases, so it mostly works as an LLM selecting from H1's top hits, not as an expansion strategy. It does behave differently on unanswerable questions: `search` was the first action on 7–8 of the 8 unanswerable questions and on 0 of the 92 answerable ones, across all budgets and both indexes. That is a useful signal for "not in the document", but it rests on just 8 questions.

**This RQ is only partly answerable with this question set.** Every question has a single gold section (`requires_all_gold` is false for all 100, and the multi-section metrics have n = 0). Expansion's main purpose, collecting evidence from several linked sections, was not tested.

| Method | Index | Gate | Hit@1 | MRR@10 | nDCG@10 (95% CI) | Evidence (context) | Context tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| T0 8k | units | on | 0.902 | 0.943 | 0.876 (0.840–0.911) | 1.000 | 5,825 | 0.04 |
| T2 8k | units | on | 0.902 | 0.943 | 0.875 (0.839–0.909) | 1.000 | 6,621 | 0.04 |
| T3 8k | units | on | 0.902 | 0.922 | 0.911 (0.870–0.949) | 0.989 | 7,784 | 0.04 |
| T2+T3 8k | units | on | 0.902 | 0.920 | 0.907 (0.863–0.947) | 0.978 | 7,839 | 0.04 |
| T4 8k | units | on | 0.978 | 0.989 | 0.916 (0.889–0.942) | 1.000 | 3,777 | 14.60 |
| T0 8k | windows | on | 0.837 | 0.910 | 0.865 (0.830–0.899) | 1.000 | 3,423 | 0.04 |
| T2 8k | windows | on | 0.837 | 0.910 | 0.865 (0.830–0.899) | 1.000 | 5,977 | 0.04 |
| T3 8k | windows | on | 0.837 | 0.889 | 0.901 (0.862–0.938) | 0.989 | 7,234 | 0.04 |
| T2+T3 8k | windows | on | 0.837 | 0.891 | 0.901 (0.862–0.938) | 1.000 | 7,501 | 0.04 |
| T4 8k | windows | on | 0.957 | 0.976 | 0.908 (0.881–0.934) | 0.978 | 2,256 | 13.11 |

Tests at the 8k budget, gate on:

| Comparison | Index | Gate | ΔHit@1 | ΔMRR@10 | ΔnDCG@10 | ΔEvidence (context) |
|---|---|---|---|---|---|---|
| T2 vs T0 | units | on | +0.000 (p=1.000) | +0.000 (p=1.000) | -0.001 (p=1.000) | +0.000 (p=1.000) |
| T2 vs T0 | windows | on | +0.000 (p=1.000) | +0.000 (p=1.000) | +0.000 (p=1.000) | +0.000 (p=1.000) |
| T3 vs T0 | units | on | +0.000 (p=1.000) | -0.021 (p=0.068) | +0.035 (p=0.117) | -0.011 (p=1.000) |
| T3 vs T0 | windows | on | +0.000 (p=1.000) | **-0.021 (p=0.029)** | +0.036 (p=0.097) | -0.011 (p=1.000) |
| T2+T3 vs T0 | units | on | +0.000 (p=1.000) | -0.023 (p=0.068) | +0.031 (p=0.368) | -0.022 (p=1.000) |
| T2+T3 vs T0 | windows | on | +0.000 (p=1.000) | -0.020 (p=0.085) | +0.036 (p=0.118) | +0.000 (p=1.000) |
| T4 vs T2+T3 | units | on | +0.076 (p=0.731) | +0.069 (p=0.070) | +0.009 (p=1.000) | +0.022 (p=1.000) |
| T4 vs T2+T3 | windows | on | **+0.120 (p=0.017)** | **+0.085 (p=0.004)** | +0.007 (p=1.000) | -0.022 (p=1.000) |

The 4k and 16k budgets and the gate-off runs are in the `significance` sheet of each `metrics.xlsx`.

## 9. RQ7: Do 500-token windows beat structure-bounded units?

**Answer:** no significant difference for any of the five signals, on section-level metrics or evidence recall. The closest is X1 nDCG, p = 0.056. The direction varies by signal: windows are nominally better for ColBERT and X1, and nominally worse for H1 on Hit@1. There is no sign that dense gains more than BM25 or ColBERT, which the plan had asked about.

**Practical reading:** windows retrieve as well as units at about half the context. With 20 hits and the gate off, H1 hands over 7,871 tokens on windows against 15,977 on units. These windows never cross a section boundary, so this result does not apply to naive fixed-size chunking.

| Signal | Hit@1 units → windows | MRR@10 units → windows | nDCG@10 units → windows | Evidence@10 units → windows | Smallest Holm p |
|---|---|---|---|---|---|
| L1 | 0.837 → 0.859 (+0.022) | 0.894 → 0.905 (+0.011) | 0.846 → 0.852 (+0.006) | 0.978 → 0.978 (+0.000) | 1.000 |
| D1 | 0.826 → 0.837 (+0.011) | 0.882 → 0.896 (+0.014) | 0.848 → 0.856 (+0.008) | 0.967 → 0.978 (+0.011) | 1.000 |
| C1 | 0.815 → 0.891 (+0.076) | 0.892 → 0.935 (+0.043) | 0.863 → 0.880 (+0.018) | 0.989 → 1.000 (+0.011) | 0.129 |
| H1 | 0.891 → 0.848 (-0.043) | 0.928 → 0.911 (-0.017) | 0.870 → 0.866 (-0.004) | 0.989 → 1.000 (+0.011) | 0.868 |
| X1 | 0.902 → 0.967 (+0.065) | 0.951 → 0.984 (+0.033) | 0.885 → 0.903 (+0.018) | 1.000 → 1.000 (+0.000) | 0.056 |

### Score gate (a design variable, not an RQ)

The gate pays for itself after fusion. Gate loss on the final H1, X1 and X2 lists is 0.000 on both indexes, and it greatly reduces cost:

| Run | Context tokens, off → on | Latency (s), off → on | Hit@1, off → on |
|---|---|---|---|
| H1 units | 15,977 → 6,276 | 0.04 → 0.04 | 0.891 → 0.902 |
| H1 windows | 7,871 → 3,430 | 0.04 → 0.04 | 0.848 → 0.837 |
| X1 windows | 8,270 → 2,619 | 2.52 → 0.49 | 0.967 → 0.967 |
| X2 units | 13,775 → 5,024 | 27.07 → 1.35 | 0.946 → 0.946 |
| X2 windows | 7,555 → 2,592 | 8.86 → 0.62 | 0.946 → 0.957 |

On a single signal it is riskier: BM25, dense, ColBERT and sparse lists each lose the gold section on 0–2% of questions compared with the same run gate-off, and Q2 loses it on 3–8%. Use the gate after fusion or reranking, and avoid relying on it with one signal alone.

## 10. Combinations to use in real projects

This is a decision made after seeing the results, on the same questions. Run A is among the top options on every metric that matters, but it is not proven to be better than the other finalists (see the table at the end of this section).

### A. Default: hybrid + bge reranker on windows, gate on

Run: `X1-win500-gateon-v1-2bebb130`.

| Stage | Setting |
|---|---|
| Chunking | 500 bge-m3 tokens, at least 50 tokens of overlap, windows never cross a section boundary |
| First stage | BM25 (k1 = 1.2, b = 0.75) and bge-m3 dense, top 50 each |
| Gate A | τ = 0.4, δ = 0.3, keep 5–50, on each list |
| Fusion | RRF, score = Σ 1 / (60 + rank) |
| Reranker | `BAAI/bge-reranker-v2-m3` on the fused list, keep at most 20 |
| Gate B | τ = 0.2, δ = 0.5, keep 5–20 |

**Results:** Hit@1 0.967, Hit@3 1.000, MRR@10 0.984, nDCG@10 0.906, evidence recall@5 1.000, about 7 hits and 2,619 tokens, 0.49 s per question, no LLM needed in retrieval.

**Why this one:** it ties for the best early precision, hands over a small context, needs no LLM at query time, and all its models have permissive licences (bge-reranker-v2-m3 is Apache-2.0, bge-m3 is MIT).

### B. If an LLM is available and you want a "no answer" signal: agent on units

Run: `T4-b8k-units-gateon-v1-34a7b4ae`. Same first stage as H1, gate on, no reranker, 8k budget.

**Results:** Hit@1 0.978, MRR@10 0.989, evidence recall in context 1.000, 3,777 tokens, about 14.6 s per question on Qwen.

Use it when 15 s is acceptable and you want the agent's behaviour on unanswerable questions. It is not significantly more accurate than A, and it depends on Qwen (or another model) acting the same way.

### C. If you have no GPU budget for a reranker: H1 on units, gate on

Run: `H1-units-gateon-v1-9f5e5fd2`.

**Results:** Hit@1 0.902, MRR@10 0.943, evidence recall@10 1.000, 6,276 tokens, 0.04 s per question. That latency includes query embedding on the GPU encoder, so expect it to be higher if bge-m3 runs on a CPU.

### About jina-reranker-v3 (X2)

It is statistically tied with X1 on windows, and on units it nominally leads X1 (Hit@1 0.946 vs 0.902). Its licence is **CC BY-NC 4.0**, so it is non-commercial unless you arrange a licence with Jina. For non-commercial work it is a valid alternative, but nothing here justifies preferring it on windows.

### Avoid or skip

| Method | Reason |
|---|---|
| Q2 keyword expansion | Significantly worse than plain BM25 |
| Q1 HyDE | About 13 s per query with no measurable gain |
| T2, T3, T2+T3 fixed expansion | No added evidence on this set; T3 variants lower MRR |
| X2, gate off, on units | 27 s and 13,775 tokens per question; turning the gate on gives the same Hit@1 and MRR |

### Finalists compared with A (exploratory)

Differences are from a paired randomisation test against A, Holm-corrected across the 8 comparisons per metric. These tests were not pre-declared.

| Method | Index | Gate | Hit@1 | MRR@10 | nDCG@10 (95% CI) | Evidence@5 | Context tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| X1 | windows | on | 0.967 | 0.984 | 0.906 (0.877–0.933) | 1.000 | 2,619 | 0.49 |
| X2 | windows | on | 0.957 | 0.978 | 0.907 (0.878–0.934) | 1.000 | 2,592 | 0.62 |
| X1 | units | on | 0.902 | 0.951 | 0.887 (0.857–0.916) | 1.000 | 5,207 | 1.34 |
| X2 | units | on | 0.946 | 0.971 | 0.900 (0.871–0.928) | 1.000 | 5,024 | 1.35 |
| X2 | units | off | 0.946 | 0.971 | 0.922 (0.896–0.947) | 1.000 | 13,775 | 27.07 |
| T4 8k | units | on | 0.978 | 0.989 | 0.916 (0.889–0.942) | 1.000 | 3,777 | 14.60 |
| T4 8k | windows | on | 0.957 | 0.976 | 0.908 (0.881–0.934) | 0.967 | 2,256 | 13.11 |
| H1 | windows | on | 0.837 | 0.910 | 0.865 (0.830–0.899) | 0.967 | 3,430 | 0.04 |
| H1 | units | on | 0.902 | 0.943 | 0.876 (0.840–0.911) | 1.000 | 6,276 | 0.04 |

| Compared with A | ΔHit@1 | ΔMRR@10 | ΔnDCG@10 | Holm p (Hit@1 / MRR / nDCG) |
|---|---|---|---|---|
| X2, windows, gate on | −0.011 | −0.005 | +0.001 | 1.000 / 1.000 / 1.000 |
| X1, units, gate on | −0.065 | −0.033 | −0.018 | 0.489 / 0.465 / 0.146 |
| X2, units, gate on | −0.022 | −0.013 | −0.005 | 1.000 / 1.000 / 1.000 |
| X2, units, gate off | −0.022 | −0.013 | +0.017 | 1.000 / 1.000 / 0.640 |
| T4 8k, units | +0.011 | +0.005 | +0.010 | 1.000 / 1.000 / 0.870 |
| T4 8k, windows | −0.011 | −0.008 | +0.002 | 1.000 / 1.000 / 1.000 |
| **H1, windows, gate on** | **−0.130** | **−0.073** | **−0.041** | **0.006 / 0.006 / 0.006** |
| H1, units, gate on | −0.065 | −0.041 | −0.029 | 0.880 / 0.465 / 0.217 |

In short, A is significantly better than its own no-reranker baseline on windows, and it cannot be separated from the other leaders.

### Not tested, so do not assume

The plan did not run a full factorial. These combinations have no results: agent + reranker, H2 + reranker, filter + reranker, and any reranker over ColBERT.

### Handling questions the document cannot answer

Retrieval always returns at least 5 hits, so it does not abstain by itself. In A, the top X1 score separates the two groups well, but this is only exploratory: all 8 unanswerable questions have a top score of 1.41 or lower, and only 5 of the 92 answerable questions do. A threshold near that value looks promising, but it has to be set on fresh data. The T4 agent's first action gives a similar signal (section 8).

## 11. Limitations to share with your advisor

1. **The question set is unreviewed and LLM-generated.** All 100 questions come from Claude (`claude-opus-5-5`), and every record has `review.status = unreviewed`. Reviewing them is the most important next step.
2. **Questions are close to the regulation's wording.** On average 80% of a question's content words appear in its gold section. Every leading method puts the gold section in the top 3 for nearly every question, so the top methods are close to a ceiling and hard to tell apart.
3. **Every question has one gold section.** Multi-section evidence, the main case for expansion and agents, was not tested.
4. **One document and 92 scored questions.** The 95% intervals on nDCG@10 are about ±0.03 for the leading methods, so small differences cannot be resolved. Results may not carry over to other documents.
5. **Picks are made on the test data.** Section 10 was chosen after seeing these questions, so the leading numbers are probably a little optimistic. No held-out set exists.
6. **Only v1 parameters were run.** The plan allows one v2 revision, which has not been made.
7. **This measures retrieval only.** Whether a generator answers correctly from these contexts was not evaluated.
8. **Latency reflects one RTX 3090 setup.** Phase A used the GPU encoder and phase B used the CPU encoder with Qwen. X1 gate-on latencies are pro-rated from the gate-off batch.

## 12. Suggested next steps

1. Have a person review the 100 questions and correct them where needed, then rerun notebook 09. If only gold labels change, rerun notebook 06 first. If question text changes, the retrieval runs need repeating too.
2. Add a held-out set of real user-style questions, with different wording and multi-section evidence, and test combination A on it. This also makes RQ4 and RQ6 properly answerable.
3. Decide, after the review, whether to use the single v2 revision the plan allows.

## 13. Where the numbers come from

| What | File |
|---|---|
| All run metrics, one row per run | `results/units/metrics.csv`, `results/win500/metrics.csv` |
| Tests, breakdowns, gate, per-question ranks, unanswerable | sheets of `results/units/metrics.xlsx` and `results/win500/metrics.xlsx` |
| Windows vs units tests | `results/granularity.csv` |
| Per-question metrics, for new paired tests | `results/question_metrics.parquet` |
| Each run's hits, stats and config | `results/runs/<run_id>/` |

The exploratory analyses are not produced by notebook 09: finalists vs A, the wording-overlap split, sibling counts, agent first actions and top-score separation. They were computed once from the files above for this report.
