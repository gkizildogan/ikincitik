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
    MP["section nDCG@10 0.876 [0.840, 0.911]<br/>Recall@10 1.000<br/>evidence recall@10 1.000<br/>kept length 5.7 · gate loss 0.033<br/>latency 0.04 s/question<br/>n = 92 answerable"]
  end
  CTX -.- MP
```

| comparison | Δ section nDCG@10 | 95% CI | p (Holm) |
|---|---|---|---|
| RQ2 H1 vs L1 | +0.043 | [+0.012, +0.076] | 0.0672 |
| RQ2 H1 vs D1 | +0.048 | [+0.015, +0.085] | 0.0672 |
| RQ2 H2 vs H1 | -0.009 | [-0.024, +0.004] | 0.9434 |
| RQ3 X1 vs H1 | +0.011 | [-0.015, +0.039] | 0.6897 |
| RQ3 X2 vs H1 | +0.024 | [-0.000, +0.050] | 0.2584 |
| RQ5 F1 vs H1 | +0.005 | [-0.007, +0.018] | 1.0000 |
| RQ5 T1 vs H1 | +0.000 | [+0.000, +0.000] | 1.0000 |
