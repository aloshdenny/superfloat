"""Tier B of the SuperFloat scaling-law study: PTQ across the Pythia ladder.

Post-training weight-only SFx quantization over 8 model sizes (70M to 12B) and
10 precisions, measured as validation loss rather than accuracy so the result
can actually be fitted. Pythia is used because it is the only open ladder with
matched data order across sizes *and* published intermediate checkpoints, which
gives the D axis for free: revision="step{n}" is the same model at a different
token count, so degradation-vs-data can be measured without training anything.

Two quantities come back from every run:

  val_loss      cross-entropy on a held-out slice, the fit target
  dead_frac     share of quantized weights at exactly zero

dead_frac is the PTQ analogue of the tier C mechanism check. Here the relevant
scale is the *trained* per-layer sigma rather than the Kaiming init, so the
collapse point should sit at a different precision than tier C predicts -- and
where it sits is the result.

GPU is measured, not assumed: modal_profile.py puts H100 at 383k tok/s for a
410m eval, the best cost per token of four candidates. Pythia-12B needs 24 GB
for fp16 weights, which rules out the A10.

Run:
    modal run modal/modal_scaling_b.py::prefetch_all   # ~49 GB, CPU, once
    modal run modal/modal_scaling_b.py --smoke
    modal run modal/modal_scaling_b.py
"""

import pathlib

import modal

BENCH_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

app = modal.App("superfloat-scaling-b")
vol = modal.Volume.from_name("sfx-baselines", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("transformers", "datasets", "numpy", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/vol/hf"})
    .add_local_dir(BENCH_DIR, remote_path="/root/sfx_bench")
)

GPU = "H100"
OUT = "/vol/runs_scaling_b"

SIZES = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
PRECISIONS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
# the D axis: same model, different token counts. 143000 is the final step.
CKPT_STEPS = [13000, 39000, 78000, 143000]
CKPT_SIZES = ["160m", "410m", "1.4b"]

EVAL_TOKENS = 400_000          # ~200 sequences of 2048; enough to separate runs
SEQLEN = 2048


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60, cpu=8)
def prefetch(model_id: str, revision: str = "main"):
    """Pull weights to the volume on CPU, so GPU time is never spent downloading."""
    from huggingface_hub import snapshot_download
    p = snapshot_download(model_id, revision=revision,
                          allow_patterns=["*.json", "*.bin", "*.safetensors",
                                          "*.txt", "*.model"])
    vol.commit()
    print(f"cached {model_id}@{revision}", flush=True)
    return p


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60, cpu=8)
def prepare_eval():
    """Tokenize a fixed held-out slice once; every run scores the same tokens."""
    import os
    import numpy as np
    if os.path.exists("/vol/scaling_b_eval.npy"):
        print("already present", flush=True)
        return
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    # datasets 5.x rejects bare ids; wikitext now lives under Salesforce/
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="test")
    ids = []
    for row in ds:
        if row["text"].strip():
            ids.extend(tok(row["text"]).input_ids)
        if len(ids) >= EVAL_TOKENS + SEQLEN:
            break
    arr = np.array(ids[: EVAL_TOKENS], dtype=np.uint16)
    np.save("/vol/scaling_b_eval.npy", arr)
    vol.commit()
    print(f"eval slice: {len(arr)} tokens", flush=True)


@app.function(image=image, gpu=GPU, volumes={"/vol": vol},
              timeout=60 * 60 * 2, max_containers=8)
def evaluate(size: str, bits: int, step: int = 0, batch: int = 4):
    """Weight-only SFx PTQ, then held-out cross-entropy."""
    import json
    import os
    import sys
    import time
    import numpy as np
    import torch
    sys.path.insert(0, "/root/sfx_bench")
    from superfloat import disable_tf32, sf_params, sf_quantize_sv

    disable_tf32()
    model_id = f"EleutherAI/pythia-{size}"
    revision = f"step{step}" if step else "main"
    tag = f"{size}_sf{bits}" + (f"_step{step}" if step else "")
    os.makedirs(OUT, exist_ok=True)

    from transformers import AutoModelForCausalLM
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.float16).cuda().eval()
    load_s = time.time() - t0

    # bits=0 is the unquantized control row
    dead = tot = 0
    if bits:
        scale, vmax = sf_params(bits)
        with torch.no_grad():
            for name, m in model.named_modules():
                # the tied embedding/output head stays in fp16, matching the
                # recipe used everywhere else in this project
                if not isinstance(m, torch.nn.Linear) or "embed_out" in name:
                    continue
                q = sf_quantize_sv(m.weight.data.float(), scale, vmax)
                dead += (q == 0).sum().item()
                tot += q.numel()
                m.weight.data = q.to(m.weight.dtype)
    torch.cuda.synchronize()

    ids = np.load("/vol/scaling_b_eval.npy").astype(np.int64)
    n_seq = len(ids) // SEQLEN
    x = torch.from_numpy(ids[: n_seq * SEQLEN]).view(n_seq, SEQLEN)

    tot_nll = tot_tok = 0.0
    with torch.no_grad():
        for i in range(0, n_seq, batch):
            xb = x[i:i + batch].cuda()
            out = model(xb, labels=xb)
            # HF averages over the batch; recover the token total
            ntok = xb.numel() - xb.shape[0]
            tot_nll += out.loss.item() * ntok
            tot_tok += ntok
    val_loss = tot_nll / tot_tok

    rec = {"size": size, "bits": bits, "step": step,
           "params": sum(p.numel() for p in model.parameters()),
           "val_loss": val_loss, "ppl": float(np.exp(val_loss)),
           "dead_frac": dead / max(tot, 1), "load_s": load_s}
    with open(f"{OUT}/{tag}.json", "w") as f:
        json.dump(rec, f)
    vol.commit()
    print(f"[{tag}] loss={val_loss:.4f} ppl={rec['ppl']:.2f} "
          f"dead={rec['dead_frac']*100:.1f}%", flush=True)
    return rec


@app.local_entrypoint()
def prefetch_all():
    jobs = [(f"EleutherAI/pythia-{s}", "main") for s in SIZES]
    jobs += [(f"EleutherAI/pythia-{s}", f"step{n}")
             for s in CKPT_SIZES for n in CKPT_STEPS]
    print(f"prefetching {len(jobs)} checkpoints", flush=True)
    list(prefetch.starmap(jobs))
    prepare_eval.remote()


@app.local_entrypoint()
def main(smoke: bool = False):
    if smoke:
        print(evaluate.remote(size="410m", bits=0))     # fp16 control
        print(evaluate.remote(size="410m", bits=4))
        return
    # bits=0 is the fp16 control row for every size
    jobs = [(s, b, 0) for s in SIZES for b in [0] + PRECISIONS]
    jobs += [(s, b, n) for s in CKPT_SIZES for b in [0] + PRECISIONS
             for n in CKPT_STEPS]
    print(f"spawning {len(jobs)} evals", flush=True)
    handles = [evaluate.spawn(size=s, bits=b, step=n) for s, b, n in jobs]
    done = 0
    for h in handles:
        try:
            h.get()
        except Exception as exc:                          # noqa: BLE001
            print(f"  eval failed: {str(exc)[:160]}", flush=True)
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{len(handles)} complete", flush=True)
