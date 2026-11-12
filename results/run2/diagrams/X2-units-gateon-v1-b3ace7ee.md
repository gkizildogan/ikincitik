# X2-units-gateon-v1-b3ace7ee

X2: H1 -> jina-reranker-v3 · index `units` · gate on · params `v1` · config `b3ace7eea425`

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
  RR["X2 jina-reranker-v3<br/>top 50 in"]
  FU --> RR
  GB{"gate B<br/>τ 0.2 δ 0.5<br/>keep 5–20"}
  RR --> GB
  CTX(["final context<br/>k ≤ 20"])
  GB --> CTX
  subgraph M["metrics"]
    MP["section nDCG@10 0.831 [0.810, 0.852]<br/>Recall@10 0.948<br/>evidence recall@10 0.945<br/>kept length 6.4 · gate loss 0.113<br/>latency 1.58 s/question<br/>n = 469 answerable"]
  end
  CTX -.- MP
```

| comparison | Δ section nDCG@10 | 95% CI | p (Holm) |
|---|---|---|---|
| RQ3 X2 vs H1 | +0.077 | [+0.060, +0.093] | 0.0006 |
| RQ3 X1 vs X2 | -0.087 | [-0.106, -0.070] | 0.0006 |
