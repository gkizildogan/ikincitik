# F1-units-gateon-v1-d85f1376

F1: H1 + LLM-extracted engine-type filter · index `units` · gate on · params `v1` · config `d85f1376506a`

```mermaid
flowchart LR
  Q(["question"])
  FL{"filter<br/>engine_type ∈ {x, all}<br/>Qwen JSON"}
  Q --> FL
  IX[("cse_units_v1<br/>base units")]
  FL --> IX
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
    MP["section nDCG@10 0.882 [0.847, 0.915]<br/>Recall@10 1.000<br/>evidence recall@10 1.000<br/>kept length 5.7 · gate loss 0.033<br/>latency 0.99 s/question<br/>n = 92 answerable"]
  end
  CTX -.- MP
```

| comparison | Δ section nDCG@10 | 95% CI | p (Holm) |
|---|---|---|---|
| RQ5 F1 vs H1 | +0.005 | [-0.007, +0.018] | 1.0000 |
