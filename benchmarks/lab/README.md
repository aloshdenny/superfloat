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
| 8 | `exp8_tierd_seeds.py` | does 4.1's inversion reproduce, with records kept? | yes: every cell within 0.016 nats |

Two further scripts target tool-use models rather than the scaling study; see
[TOOL_USE_QAT.md](../../TOOL_USE_QAT.md).

| script | question | answer |
| --- | --- | --- |
| `stage0_toolqat.py` | does SF survive RMSNorm/SwiGLU/GQA from scratch? | SF8 free, SF4 usable under absorption |
| `smol_qat.py` | can a trained checkpoint be moved onto the SF grid? | SF8 needs no training at all |
| `bfcl_eval.py` | does PTQ preserve BFCL, and does that hold at scale? | SF8 free from 1.7B; SF6 cliff is small models |
| `train_1b.py` | does from-scratch SF8 QAT hold at Llama-3.2-1B? | in flight; no archive yet |

A pure-SF datapath, not weights-only: [PURE_SF.md](../../PURE_SF.md).

| script | question | answer |
| --- | --- | --- |
| `psd.py` | what scale granularity keeps saturate-every-register SF8 alive? | per-token; tensor/channel do not |

Domain evals: [DOMAIN.md](../../DOMAIN.md).

| script | question | answer |
| --- | --- | --- |
| `vision_ptq.py` | YOLO-seg and box-prompt SAM under weights-only SF? | YOLO mask mAP 0.40 → 0.0; SAM IoU ~0.46 |
| `code_xlat.py` | does C→C++ still compile under SF PTQ? | runner fixed; no number yet |
| `sam_scratch.py` | from-scratch box→mask QAT, TinyBoxSeg on coco128-seg | no archive yet |
| `sam_pretrain.py` | same idea, BoxSeg-S on COCO train2017 | no archive yet |

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
python exp8_tierd_seeds.py --prepare       # same corpus as exp3
python exp8_tierd_seeds.py --size 11m --tpp 10 --bits 2 --seed 1
python exp3_11m.py --prepare               # 11M D/N under absorption (open 4.1)
python exp3_11m.py --queue                 # 4GB-safe: micro 1, accum 16, ckpt
python exp_ptq_absorb.py --prepare
python exp_ptq_absorb.py --queue           # Pythia PTQ ± absorption, 4GB ok
python bfcl_eval.py --model Qwen/Qwen3-1.7B --bits 8 --mode ln_all
python vision_ptq.py --task both --bits 8
python psd.py --ptq
```

## Concurrency

These pods have 24 GB. A CIFAR run holds about 3 GB and six fit comfortably;
the GPU is launch-bound below that and sits at 20% utilisation with one stream.
The 11M model in `exp8_tierd_seeds.py` allocates about 13.4 GB, so three
streams fit a 46 GB card and a fourth OOMs. The 5M language model in
`exp3_reg.py` is different: it allocates about 8.5 GB
but its caching allocator reserves up to 14.7 GB, so **two streams are the
limit and three OOM**. Run it with

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

which stops the reserved pool from growing past what is allocated. A queue that
ends with a sequential pass over the full configuration list will pick up
anything that still died.
