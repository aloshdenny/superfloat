# Domain evals: segmentation, promptable SAM, C→C++

The tool-use document answers whether SF preserves function calling. The
paper also needs a known-class dense predictor, a promptable unknown-object
segmenter, and a public proxy for the fighter-jet C→C++ claim. This is that
sweep. Weights-only PTQ, `ln_all` where the model is Llama-shaped, head left
in fp32.

---

## 1. The headline

**YOLO11n-seg dies under SF PTQ. Box-prompt SAM ViT-B does not.** Same
coco128-seg images, same 24 GB card, same weights-only recipe.

| task | fp32 / identity | SF8 | SF6 |
| --- | --- | --- | --- |
| YOLO11n-seg mask mAP | 0.399 | **0.0** | **0.0** |
| YOLO11n-seg box mAP | 0.488 | **0.0** | **0.0** |
| SAM ViT-B mean IoU vs fp32 teacher | 1.0 | 0.463 | 0.449 |

The YOLO collapse matches the M4 Air run (fp32 mask 0.39, SF8/SF6 0.0), so
it is not a CUDA-vs-MPS artefact. SAM is PTQ fidelity against the same
model's own fp32 box-prompt masks on 16 images, not a COCO-trained SAM
leaderboard number. Identity bits=0 is 1.0 by construction.

A known-class detector with no per-channel scale on the backbone dies. A
promptable ViT encoder, asked only to match itself, loses about half its
IoU and then stops getting worse from SF8 to SF6.

The C→C++ proxy (below) adds a third shape to this picture: **SF8 nearly
matches bf16, then SF6 collapses**, a one-step-later cliff than YOLO's but
the same qualitative story -- PTQ holds until it doesn't, and where it
breaks is task/architecture-specific, not a fixed bit count.

| C→C++ compile-pass, Qwen3-0.6B, `ln_all` | bf16 | SF8 | SF6 |
| --- | --- | --- | --- |
| pass rate (n=30) | 86.7% | 83.3% | **26.7%** |

---

## 2. What this is not

- **Not QAT.** The scaling study and the tool-use document both found PTQ
  cliffs that QAT later recovered. YOLO-seg is in that state: the cliff is
  measured, the recovery is not. Channel-scale absorption on the backbone
  was not tried either.
- **Not SA-1B, not SAM-H.** Real SAM-B/H pretraining does not fit a 16 GB
  Mac or a $0.27/hr 3090. `sam_scratch.py` / `sam_pretrain.py` are the
  runnable stand-in: class-agnostic box→mask on coco128-seg (TinyBoxSeg,
  ~1.1M) and on COCO train2017 (BoxSeg-S, ~18M). 16 GB unified memory
  jetsams mixed GPU+CPU training and dual-ViT SAM loads; the from-scratch
  probe is CPU 2-wide or nothing. Both now have archived results
  (`domain_sam_pretrain.jsonl`, `domain_coco128_seeds.jsonl`):

  | model | bits | val IoU |
  | --- | --- | --- |
  | BoxSeg-S (COCO, 6 epochs) | SF16 | 0.830 |
  | BoxSeg-S (COCO, 6 epochs) | SF8 | 0.832 |
  | BoxSeg-S (COCO, 6 epochs) | SF4 | 0.797 |
  | TinyBoxSeg (coco128, 40 ep, 2-3 seeds) | fp32/fp16 | 0.53-0.60 |
  | TinyBoxSeg (coco128, 40 ep, 2 seeds) | SF16 | 0.58-0.59 |
  | TinyBoxSeg (coco128, 40 ep, 2 seeds) | SF8 | 0.585 |
  | TinyBoxSeg (coco128, 2-3 seeds) | SF4 | 0.23-0.51, one seed only 9/40 epochs |

  From-scratch training at SF8/SF16 tracks fp32/fp16 within seed noise on
  both scales; SF4 is where it starts to degrade, and on TinyBoxSeg one
  SF4 seed did not finish (9/40 epochs, not re-run) so that row is noisier
  than the others, not a clean third point.
- **C→C++ is a proxy, not a fighter-jet number.** `code_xlat.py`: short C
  functions must come back as C++ that `clang++ -fsyntax-only` accepts,
  scored as compile-pass rate on 30 fixed snippets, not BLEU, not a real
  avionics codebase. The first 0.6B bf16 sample crashed the runner
  (`(r.stderr or "")[-400]` indexed a character on empty stderr, fixed to
  `[-400:]`) and no json was written at the time; that run has since
  completed cleanly (`code_xlat.jsonl`, table above). 30 snippets is small
  -- individual pass/fail flips are a couple of samples, not a tight CI,
  though the SF6 cliff (8/30 vs SF8's 25/30) is far past that noise floor.

---

## 3. Files

```
benchmarks/lab/
  vision_ptq.py     YOLO11n-seg val + SAM ViT-B box-prompt IoU
  code_xlat.py      C→C++ compile-pass proxy
  sam_scratch.py    TinyBoxSeg from-scratch SF2–SF16 on coco128-seg
  sam_pretrain.py   BoxSeg-S on COCO train2017 (needs local COCO)
  run_domain.sh     interrupt-safe queue, mac | pod
  train_yolo.py     YOLO QAT; --format accepts any sfx, --device settable
benchmarks/results/
  yolo_seg.jsonl              3 arms, section 1
  sam_box.jsonl                3 arms, section 1
  code_xlat.jsonl              3 arms (bf16/sf8/sf6), section 1
  domain_sam_pretrain.jsonl    BoxSeg-S, 3 bit-widths, section 2
  domain_coco128_seeds.jsonl   TinyBoxSeg, 2-3 seeds x 4 bit-widths, section 2
```

```bash
cd benchmarks/lab
python vision_ptq.py --task both --bits 8
python code_xlat.py --model Qwen/Qwen3-0.6B --bits 8
bash run_domain.sh pod
```

Live per-config JSON still lands under `benchmarks/results/domain/` so a
queue can skip finished arms. The repository tracks the folded jsonl,
not the loose files, not `*.pt`, not Ultralytics `yolo_runs/`.
