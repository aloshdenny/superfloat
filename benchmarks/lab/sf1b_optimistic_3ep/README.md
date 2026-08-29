# Optimistic 3-epoch ppl / loss projection

Log–log regression on the last 100 evals of the combined 4090+H100
`metrics.jsonl` (0.88B–1.20B tokens):

```
val_ppl  = A · T^b
val_loss = log(val_ppl)
A = 5284.27
b = −0.2588
in-sample RMSE = 0.05 ppl
```

Cadence matches the live 4090 recipe: log every 50 steps, 32768 tokens/step.
First row of the jsonl is the last observed eval; the rest is the projection
through **60B tokens (3 epochs of the 20B FineWeb-Edu shard)**.

| tokens | epochs | val ppl | val loss |
|--------|--------|---------|----------|
| 1.20B (now) | 0.06 | 23.56 | 3.16 |
| 20B | 1 | 11.41 | 2.43 |
| 40B | 2 | 9.53 | 2.25 |
| 60B | 3 | 8.58 | 2.15 |

Plot: `tokens_epochs_ppl_loss.svg` (and `.png`) — continuous curve, training
tokens / epochs vs perplexity and loss only.

This is the **no-floor optimistic** continuation of the recent slope. It will
overshoot if the exponent keeps decaying.
