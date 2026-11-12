# C1-units-gateoff-v1-b814c18d

C1: ColBERT MaxSim over all units · index `units` · gate off · params `v1` · config `b814c18d8312`

```mermaid
flowchart LR
  Q(["question"])
  IX[("cse_units_v1<br/>base units")]
  Q --> IX
  S0["ColBERT MaxSim<br/>client, exact<br/>top-50"]
  IX --> S0
  CTX(["final context<br/>k ≤ 20"])
  S0 --> CTX
  subgraph M["metrics"]
    MP["section nDCG@10 0.752 [0.727, 0.777]<br/>Recall@10 0.934<br/>evidence recall@10 0.908<br/>latency 0.81 s/question<br/>n = 469 answerable"]
  end
  CTX -.- MP
```

| comparison | Δ section nDCG@10 | 95% CI | p (Holm) |
|---|---|---|---|
| RQ1 C1 vs L1 | +0.091 | [+0.062, +0.120] | 0.0006 |
| RQ7 C1@win500 vs C1@units | +0.014 | [+0.002, +0.026] | 0.0592 |
