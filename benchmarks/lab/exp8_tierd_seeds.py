"""Experiment 8: restore the three-seed evidence for section 4.1.

Tier D reported the ordering inverted under scale absorption at 11m -- SF2
(+0.108) beating SF3 (+0.152) beating SF4 (+0.187), consistently across three
seeds. Those seed replicates ran on ephemeral capacity and were never written
back, so the archive holds only the 36-run base grid and the claim survives in
a commit message rather than in data.

This reruns the whole cell self-contained: bits {0,2,3,4} x seeds {0,1,2} at
11m under mode `ln`, including its own FP32 controls, so the penalties do not
depend on a control measured elsewhere. Absolute losses will not match tier D,
because the validation split here is the tail of a 400M-token corpus rather
than of Modal's 1.1B one. The penalties are the comparable quantity, and they
are what 4.1 reports.

exp3 already showed this architecture reproduces across hardware at 5m, within
0.017 nats. What is untested is whether the inversion itself reproduces.
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from superfloat import disable_tf32, apply_superfloat, SFLinear, sf_quantize_sv

VOCAB, SEQLEN = 50304, 1024
OUT = "/workspace/results"
TOKENS = "/workspace/fineweb_edu_tokens.bin"
TARGET_TOKENS = 400_000_000          # 40x of the 5m model is 190M; headroom
CONFIGS = {"5m": (256, 6, 4), "11m": (384, 6, 6)}   # as tier D
D_MODEL, N_LAYER, N_HEAD = CONFIGS["11m"]


def prepare_tokens():
    if os.path.exists(TOKENS):
        p = np.memmap(TOKENS, dtype=np.uint16, mode="r")
        if len(p) >= TARGET_TOKENS and p[-4096:].max() > 0:
            print(f"corpus present: {len(p)/1e6:.0f}M", flush=True); return
        del p; os.remove(TOKENS)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2"); eot = tok.eos_token_id
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    buf = np.memmap(TOKENS, dtype=np.uint16, mode="w+", shape=(TARGET_TOKENS,))
    i = 0; batch = []
    def flush(b, i):
        for ids in tok(b)["input_ids"]:
            ids = ids + [eot]
            if i + len(ids) > TARGET_TOKENS: ids = ids[:TARGET_TOKENS - i]
            if not ids: break
            buf[i:i+len(ids)] = np.array(ids, dtype=np.uint16); i += len(ids)
        return i
    for row in ds:
        batch.append(row["text"])
        if len(batch) >= 2000:
            i = flush(batch, i); batch = []
            if i >= TARGET_TOKENS: break
    if i < TARGET_TOKENS: i = flush(batch, i)
    buf.flush(); del buf
    print(f"corpus written: {i/1e6:.0f}M", flush=True)


def build():
    class Attn(nn.Module):
        def __init__(s):
            super().__init__()
            s.q=nn.Linear(D_MODEL,D_MODEL); s.k=nn.Linear(D_MODEL,D_MODEL)
            s.v=nn.Linear(D_MODEL,D_MODEL); s.proj=nn.Linear(D_MODEL,D_MODEL)
        def forward(s,x):
            B,T,C=x.shape; hd=C//N_HEAD
            q=s.q(x).view(B,T,N_HEAD,hd).transpose(1,2)
            k=s.k(x).view(B,T,N_HEAD,hd).transpose(1,2)
            v=s.v(x).view(B,T,N_HEAD,hd).transpose(1,2)
            o=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            return s.proj(o.transpose(1,2).reshape(B,T,C))
    class Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.ln1=nn.LayerNorm(D_MODEL); s.attn=Attn(); s.ln2=nn.LayerNorm(D_MODEL)
            s.mlp=nn.Sequential(nn.Linear(D_MODEL,4*D_MODEL),nn.GELU(),
                                nn.Linear(4*D_MODEL,D_MODEL))
        def forward(s,x):
            x=x+s.attn(s.ln1(x)); return x+s.mlp(s.ln2(x))
    class GPT(nn.Module):
        def __init__(s):
            super().__init__()
            s.wte=nn.Embedding(VOCAB,D_MODEL); s.wpe=nn.Embedding(SEQLEN,D_MODEL)
            s.blocks=nn.ModuleList([Block() for _ in range(N_LAYER)])
            s.lnf=nn.LayerNorm(D_MODEL); s.head=nn.Linear(D_MODEL,VOCAB,bias=False)
            s.head.weight=s.wte.weight
        def forward(s,idx):
            x=s.wte(idx)+s.wpe(torch.arange(idx.shape[1],device=idx.device))
            for b in s.blocks: x=b(x)
            return s.head(s.lnf(x))
    return GPT()


def install_col_norm(model):
    """Per-input-channel scale, absorbed by the norm feeding each matmul.
    q/k/v share one g because they all read ln1."""
    class SFLinearCol(SFLinear):
        sf_group = None
        def forward(self, x):
            w = self.weight
            src = self.sf_group if self.sf_group is not None else [w]
            g = torch.stack([t.abs().amax(dim=0) for t in src]).amax(0).clamp_min(1e-8)
            wq = sf_quantize_sv(w / g.unsqueeze(0), self.sf_scale, self.sf_vmax)
            b = None if self.bias is None else sf_quantize_sv(self.bias, self.sf_scale, self.sf_vmax)
            return F.linear(x * g, wq, b)
    n = 0
    for blk in model.blocks:
        grp = [blk.q.weight, blk.k.weight, blk.v.weight] if hasattr(blk,"q") else \
              [blk.attn.q.weight, blk.attn.k.weight, blk.attn.v.weight]
        at = blk.attn
        for m in (at.q, at.k, at.v):
            m.__class__ = SFLinearCol; m.sf_group = grp; n += 1
        blk.mlp[0].__class__ = SFLinearCol; blk.mlp[0].sf_group = None; n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--tpp", type=int, default=10, help="tokens per param")
    ap.add_argument("--size", default="11m", choices=sorted(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--prepare", action="store_true")
    a = ap.parse_args()
    if a.prepare: prepare_tokens(); return

    global D_MODEL, N_LAYER, N_HEAD
    D_MODEL, N_LAYER, N_HEAD = CONFIGS[a.size]
    disable_tf32(); torch.manual_seed(a.seed)
    tag = f"exp8_{a.size}_ln_" + ("fp32" if a.bits==0 else f"sf{a.bits}") + f"_s{a.seed}"
    if os.path.exists(f"{OUT}/{tag}.json"):
        print(f"[{tag}] done, skip", flush=True); return
    model = build().cuda()
    n_ne = sum(p.numel() for p in model.parameters()) \
           - model.wte.weight.numel() - model.wpe.weight.numel()
    ncol = 0
    if a.bits:
        nconv = apply_superfloat(model, bits=a.bits, head_names=("head","wte","wpe"),
                                 quantize_activations=False)
        assert nconv == 6*N_LAYER, f"quantized {nconv}, expected {6*N_LAYER}"
        ncol = install_col_norm(model)
    total = int(n_ne * a.tpp); steps = total // (a.batch * SEQLEN)
    print(f"[{tag}] N={n_ne/1e6:.2f}M tpp={a.tpp} tokens={total/1e6:.0f}M steps={steps} col={ncol}", flush=True)

    data = np.memmap(TOKENS, dtype=np.uint16, mode="r")
    assert len(data) >= TARGET_TOKENS and data[-4096:].max() > 0, "corpus incomplete"
    train_end = len(data) - 2_000_000
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9,0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.02)
    lf = nn.CrossEntropyLoss(); rng = np.random.default_rng(a.seed)
    def get(lo,hi):
        ix=rng.integers(lo,hi-SEQLEN-1,size=a.batch)
        x=np.stack([data[i:i+SEQLEN] for i in ix]).astype(np.int64)
        y=np.stack([data[i+1:i+1+SEQLEN] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
    hist,t0=[],time.time()
    for step in range(steps):
        x,y=get(0,train_end); opt.zero_grad(set_to_none=True)
        lf(model(x).view(-1,VOCAB),y.view(-1)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sched.step()
        if step%max(1,steps//10)==0 or step==steps-1:
            model.eval(); vl=k=0.0
            with torch.no_grad():
                for _ in range(20):
                    xv,yv=get(train_end,len(data))
                    vl+=lf(model(xv).view(-1,VOCAB),yv.view(-1)).item(); k+=1
            model.train(); hist.append({"step":step,"val_loss":vl/k})
    rec={"exp":"exp8","size":a.size,"mode":"ln","bits":a.bits,"tpp":a.tpp,
         "seed":a.seed,"n_nonembed":n_ne,
         "tokens":total,"steps":steps,"col_norm_layers":ncol,
         "final_val_loss":hist[-1]["val_loss"],"minutes":(time.time()-t0)/60,
         "history":hist,"complete":True}
    os.makedirs(OUT,exist_ok=True); json.dump(rec,open(f"{OUT}/{tag}.json","w"))
    print(f"[{tag}] DONE val={hist[-1]['val_loss']:.4f} ({rec['minutes']:.0f}m)", flush=True)


if __name__ == "__main__":
    main()
