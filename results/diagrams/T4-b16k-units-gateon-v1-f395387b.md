# T4-b16k-units-gateon-v1-f395387b

T4: H1 -> agentic small-to-big · index `units` · gate on · params `v1` · config `f395387b14d4`

```mermaid
flowchart LR
  Q(["question"])
  IX[("cse_units_v1<br/>base units")]
  Q --> IX
  S0["BM25<br/>top-50"]
  IX --> S0
  GA0{"gate A<br/>τ 0.4 δ 0.3<br/>keep 5–50"}
  S0 --> GA0
  S1["bge-m3 dense<br/>top-50"]
  IX --> S1
  GA1{"gate A<br/>τ 0.4 δ 0.3<br/>keep 5–50"}
  S1 --> GA1
  FU["RRF<br/>k = 60"]
  GA0 --> FU
  GA1 --> FU
  EX["T4 agent<br/>≤5 steps, budget 16384<br/>expand · follow_refs · search · done"]
  FU --> EX
  CTX(["final context<br/>budgeted"])
  EX --> CTX
  subgraph M["metrics"]
    MP["section nDCG@10 0.913 [0.886, 0.938]<br/>Recall@10 1.000<br/>evidence recall@10 1.000<br/>kept length 5.7 · gate loss 0.033<br/>latency 15.38 s/question<br/>n = 92 answerable"]
  end
  CTX -.- MP
```

| comparison | Δ section nDCG@10 | 95% CI | p (Holm) |
|---|---|---|---|
| RQ6 T4 vs T2+T3 | -0.001 | [-0.039, +0.040] | 1.0000 |
