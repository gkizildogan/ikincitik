# H1-win500-gateoff-v1-9d5dfb15

H1: BM25 + dense, RRF · index `win500` · gate off · params `v1` · config `9d5dfb150d1e`

```mermaid
flowchart LR
  Q(["question"])
  IX[("cse_win500_v1<br/>500-token windows")]
  Q --> IX
  S0["BM25<br/>top-50"]
  IX --> S0
  S1["bge-m3 dense<br/>top-50"]
  IX --> S1
  FU["RRF<br/>k = 60"]
  S0 --> FU
  S1 --> FU
  CTX(["final context<br/>k ≤ 20"])
  FU --> CTX
  subgraph M["metrics"]
    MP["section nDCG@10 0.739 [0.713, 0.766]<br/>Recall@10 0.928<br/>evidence recall@10 0.886<br/>latency 0.02 s/question<br/>n = 469 answerable"]
  end
  CTX -.- MP
```

| comparison | Δ section nDCG@10 | 95% CI | p (Holm) |
|---|---|---|---|
| RQ2 H1 vs L1 | +0.074 | [+0.058, +0.090] | 0.0008 |
| RQ2 H1 vs D1 | -0.016 | [-0.036, +0.004] | 0.5829 |
| RQ2 H2 vs H1 | -0.004 | [-0.015, +0.007] | 0.9267 |
| RQ3 X1 vs H1 | +0.057 | [+0.038, +0.076] | 0.0006 |
| RQ3 X2 vs H1 | +0.125 | [+0.104, +0.146] | 0.0006 |
| RQ5 F1 vs H1 | +0.024 | [+0.012, +0.036] | 0.0008 |
| RQ5 T1 vs H1 | +0.004 | [+0.001, +0.007] | 0.0114 |
| RQ7 H1@win500 vs H1@units | -0.005 | [-0.015, +0.005] | 0.6617 |
