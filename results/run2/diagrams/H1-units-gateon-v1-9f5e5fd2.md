# H1-units-gateon-v1-9f5e5fd2

H1: BM25 + dense, RRF · index `units` · gate on · params `v1` · config `9f5e5fd2f264`

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
  CTX(["final context<br/>k ≤ 20"])
  FU --> CTX
  subgraph M["metrics"]
    MP["section nDCG@10 0.755 [0.730, 0.779]<br/>Recall@10 0.946<br/>evidence recall@10 0.937<br/>kept length 6.2 · gate loss 0.113<br/>latency 0.02 s/question<br/>n = 469 answerable"]
  end
  CTX -.- MP
```

| comparison | Δ section nDCG@10 | 95% CI | p (Holm) |
|---|---|---|---|
| RQ2 H1 vs L1 | +0.113 | [+0.095, +0.132] | 0.0008 |
| RQ2 H1 vs D1 | +0.029 | [+0.012, +0.047] | 0.0096 |
| RQ2 H2 vs H1 | -0.009 | [-0.020, +0.001] | 0.2968 |
| RQ3 X1 vs H1 | -0.011 | [-0.028, +0.006] | 0.4216 |
| RQ3 X2 vs H1 | +0.077 | [+0.060, +0.093] | 0.0006 |
| RQ5 F1 vs H1 | +0.016 | [+0.004, +0.027] | 0.0264 |
| RQ5 T1 vs H1 | +0.004 | [+0.001, +0.008] | 0.0460 |
