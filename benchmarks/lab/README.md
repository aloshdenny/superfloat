# Follow-up experiments

Seven experiments run after the four-tier study, on single RTX A5000 / RTX 3090
pods rather than Modal. They ask where the precision that the tier study did
not explain actually goes.

Each script writes one JSON per configuration to `$OUT` and skips a
configuration whose result file already exists, so a queue can be interrupted
and restarted without losing or repeating work. Every sweep carries an
unquantized control trained in the same condition; the controls are what make a
silent bug visible, and several were caught that way.

| # | script | question | answer |
| --- | --- | --- | --- |
| 1 | `exp1_act.py` | how do weight and activation precision trade off? | asymmetric: weights need 3-4 bits, activations 6 |
| 2 | `exp2_dn.py` | how does PTQ damage move with training tokens? | U-shaped; both ends of a run are fragile |
| 3 | `exp3_reg.py` | how does the penalty move with tokens-per-parameter? | U-shaped; decay is not the whole story |
| 4 | `exp1_act.py --depth` | does depth move the critical precision? | no; it lowers the penalty |
| 5 | `exp5_alloc.py` | where do dead weights sit, layer by layer? | in wide layers, exactly as fan_in predicts |
| 6 | `exp6_lr.py` | does usable step size track grid resolution? | no, not over a 640x range |
| 7 | `exp1_act.py --no-chan-norm --depth` | does the paper's ResNet-56 instability reproduce? | no: 0.94 pp against 12.0 |

## Running

`exp1_act.py` supplies the CIFAR-100 model and data loader that experiments 4,
6 and 7 import, so it must sit beside them, along with `superfloat.py` from the
parent directory.

```bash
python exp1_act.py  --bits-w 4 --bits-a 6 --depth 20 --seed 0
python exp2_dn.py   --size 410m --step 39000 --bits 7
python exp3_reg.py  --prepare              # tokenise once, ~10 min
python exp3_reg.py  --tpp 20 --bits 4 --seed 0
python exp5_alloc.py --bits 4
python exp6_lr.py   --bits 8 --lr 4e-3
```

## Concurrency

These pods have 24 GB. A CIFAR run holds about 3 GB and six fit comfortably;
the GPU is launch-bound below that and sits at 20% utilisation with one stream.
The 5M language model in `exp3_reg.py` is different: it allocates about 8.5 GB
but its caching allocator reserves up to 14.7 GB, so **two streams are the
limit and three OOM**. Run it with

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

which stops the reserved pool from growing past what is allocated. A queue that
ends with a sequential pass over the full configuration list will pick up
anything that still died.
