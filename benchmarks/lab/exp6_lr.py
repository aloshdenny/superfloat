"""Experiment 6: the learning-rate law under quantization.

Observed early in the study but never formalised: SF8 from random init diverged
at the FP32 recipe's 4e-3 and trained at 1e-3, while SF16 tolerated 4e-3. If the
usable step size is set by grid resolution, then

    eta*(p) ~ 2^-p   (one fewer bit halves the usable learning rate)

Short runs suffice: divergence shows up in the first few epochs, so this maps
the (p, lr) stability boundary cheaply rather than training every cell out.
Reported as the largest lr that still beats chance by a clear margin.
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from superfloat import disable_tf32, apply_superfloat, clamp_all
from exp1_act import build, load_gpu, MEAN, STD

OUT = "/workspace/results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    disable_tf32(); torch.manual_seed(a.seed)
    tag = f"exp6_sf{a.bits}_lr{a.lr:.0e}_s{a.seed}"
    if os.path.exists(f"{OUT}/{tag}.json"):
        print(f"[{tag}] done, skip", flush=True); return

    model = build().cuda()
    apply_superfloat(model, bits=a.bits, head_names=("head",),
                     quantize_activations=False)
    xtr, ytr = load_gpu(True); xte, yte = load_gpu(False)
    mean = torch.tensor(MEAN, device="cuda").view(1,3,1,1)
    std = torch.tensor(STD, device="cuda").view(1,3,1,1)
    norm = lambda t: (t.float().div(255.0)-mean)/std
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    lf = nn.CrossEntropyLoss()
    accs, diverged = [], False
    t0=time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(xtr), device="cuda")
        for i in range(0, len(xtr)-a.batch+1, a.batch):
            idx = perm[i:i+a.batch]
            opt.zero_grad(set_to_none=True)
            loss = lf(model(norm(xtr[idx])), ytr[idx])
            if not torch.isfinite(loss): diverged = True; break
            loss.backward(); opt.step(); clamp_all(model)
        if diverged: break
        model.eval(); c=t=0
        with torch.no_grad():
            for i in range(0,len(xte),500):
                out=model(norm(xte[i:i+500]))
                c+=(out.argmax(1)==yte[i:i+500]).sum().item(); t+=len(out)
        accs.append(100.0*c/t)
    best = max(accs) if accs else 0.0
    rec = {"exp":"exp6","bits":a.bits,"lr":a.lr,"seed":a.seed,"best_acc":best,
           "diverged":diverged or best < 2.0,   # 100 classes: chance is 1%
           "accs":accs,"minutes":(time.time()-t0)/60,"complete":True}
    os.makedirs(OUT,exist_ok=True); json.dump(rec, open(f"{OUT}/{tag}.json","w"))
    print(f"[{tag}] best={best:.2f} diverged={rec['diverged']}", flush=True)


if __name__ == "__main__":
    main()
