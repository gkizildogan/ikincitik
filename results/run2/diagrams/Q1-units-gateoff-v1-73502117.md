# Q1-units-gateoff-v1-73502117

Q1: HyDE (dense) · index `units` · gate off · params `v1` · config `73502117bcc2`

```mermaid
flowchart LR
  Q(["question"])
  QT["Q1 HyDE<br/>Qwen, 1 passage,<br/>mean with question"]
  Q --> QT
  IX[("cse_units_v1<br/>base units")]
  QT --> IX
  S0["bge-m3 dense<br/>top-50"]
  IX --> S0
  CTX(["final context<br/>k ≤ 20"])
  S0 --> CTX
  subgraph M["metrics"]
    MP["section nDCG@10 0.780 [0.757, 0.802]<br/>Recall@10 0.964<br/>evidence recall@10 0.945<br/>latency 12.98 s/question<br/>n = 469 answerable"]
  end
  CTX -.- MP
```

| comparison | Δ section nDCG@10 | 95% CI | p (Holm) |
|---|---|---|---|
| RQ4 Q1 vs D1 | +0.031 | [+0.017, +0.047] | 0.0008 |
