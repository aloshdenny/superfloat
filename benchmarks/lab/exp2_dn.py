"""Experiment 2: the D/N precision law.

Tier B found PTQ damage grows with training tokens, and that the effect weakens
with model size (10x at 160m, 4.5x at 410m, 1.5x at 1.4b). That pattern is a
tokens-per-parameter story sampled at only 4 checkpoints x 3 sizes.

Pythia publishes 143 checkpoints per model, so the data axis is free. This
sweeps a denser grid and fits

    p_min(D/N) = minimum precision holding degradation under a threshold

Prediction: p_min grows roughly logarithmically in D/N, and the U-shape tier B
saw across model sizes collapses onto one curve when replotted against D/N.

    python exp2_dn.py --size 410m --step 39000 --bits 6
    python exp2_dn.py --prefetch          # CPU only, no GPU needed
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from superfloat import disable_tf32, sf_params, sf_quantize_sv

OUT = "/workspace/results"
EVAL = "/workspace/eval_tokens.npy"
SEQLEN = 2048
SIZES = ["160m", "410m", "1.4b"]
# log-spaced through training; 143000 is the final checkpoint (300B tokens)
STEPS = [1000, 3000, 8000, 20000, 39000, 78000, 143000]
PRECS = [4, 5, 6, 7, 8, 10, 16]
TOK_PER_STEP = 2097152           # Pythia batch: 1024 seq x 2048 tokens


def prepare_eval():
    if os.path.exists(EVAL):
        print("eval slice present", flush=True); return
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    ids = []
    for row in ds:
        if row["text"].strip():
            ids.extend(tok(row["text"]).input_ids)
        if len(ids) >= 400_000: break
    np.save(EVAL, np.array(ids[:400_000], dtype=np.uint16))
    print(f"eval slice: {len(ids[:400_000])} tokens", flush=True)


def prefetch():
    """CPU-only: pull every checkpoint so GPU time is never spent downloading."""
    from huggingface_hub import snapshot_download
    prepare_eval()
    for s in SIZES:
        for st in STEPS:
            try:
                snapshot_download(f"EleutherAI/pythia-{s}", revision=f"step{st}",
                                  allow_patterns=["*.json", "*.bin", "*.safetensors",
                                                  "*.txt"])
                print(f"cached {s}@step{st}", flush=True)
            except Exception as e:
                print(f"FAILED {s}@step{st}: {str(e)[:120]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="410m"); ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--bits", type=int, default=6); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--prefetch", action="store_true")
    a = ap.parse_args()
    if a.prefetch: prefetch(); return

    disable_tf32()
    tag = f"exp2_{a.size}_step{a.step}_sf{a.bits}"
    if os.path.exists(f"{OUT}/{tag}.json"):
        print(f"[{tag}] done, skip", flush=True); return
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{a.size}", revision=f"step{a.step}",
        dtype=torch.float16).cuda().eval()

    dead = tot = 0
    if a.bits:
        HEAD = ("embed_out", "lm_head")
        heads = [n for n, m in model.named_modules()
                 if isinstance(m, torch.nn.Linear) and n.rsplit(".", 1)[-1] in HEAD]
        if len(heads) != 1:
            raise RuntimeError(f"expected one head, found {heads}")
        scale, vmax = sf_params(a.bits)
        with torch.no_grad():
            for n, m in model.named_modules():
                if not isinstance(m, torch.nn.Linear): continue
                q = sf_quantize_sv(m.weight.data.float(), scale, vmax)
                dead += (q == 0).sum().item(); tot += q.numel()
                m.weight.data = q.to(m.weight.dtype)
    torch.cuda.synchronize()

    ids = np.load(EVAL).astype(np.int64)
    n_seq = len(ids) // SEQLEN
    x = torch.from_numpy(ids[:n_seq * SEQLEN]).view(n_seq, SEQLEN)
    nll = ntok = 0.0
    with torch.no_grad():
        for i in range(0, n_seq, a.batch):
            xb = x[i:i + a.batch].cuda()
            out = model(xb, labels=xb)
            k = xb.numel() - xb.shape[0]
            nll += out.loss.item() * k; ntok += k
    loss = nll / ntok
    rec = {"exp": "exp2", "size": a.size, "step": a.step, "bits": a.bits,
           "tokens": a.step * TOK_PER_STEP,
           "params": sum(p.numel() for p in model.parameters()),
           "val_loss": loss, "dead_frac": dead / max(tot, 1),
           "minutes": (time.time() - t0) / 60, "complete": True}
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(f"{OUT}/{tag}.json", "w"))
    print(f"[{tag}] loss={loss:.4f} dead={100*rec['dead_frac']:.1f}% "
          f"tokens={rec['tokens']/1e9:.0f}B", flush=True)


if __name__ == "__main__":
    main()
