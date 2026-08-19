"""Does SF quantization preserve tool-calling *capability*, not just loss?

Every number in TOOL_USE_QAT.md is a validation loss. Loss is not capability:
a model can lose 0.1 nats and still call the right function every time, or gain
nothing and start emitting malformed JSON. This runs the Berkeley Function
Calling Leaderboard against the same model in bf16 and in SF, and scores the
generated call by AST match the way BFCL does.

Three categories, chosen because they need no live execution:

  simple       one function available, one correct call
  multiple     several functions available, must pick the right one
  irrelevance  no function applies, and the model must NOT call one

The third is the interesting one for a deployed agent. A model that calls
something plausible on every prompt scores well on the first two and is
useless in practice.

Weights only, as everywhere else in this repository. Biases and norms stay in
bf16; they sit outside the systolic array.
"""
import argparse, json, os, re, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smol_qat import install, fold          # same surgery, Qwen2 is Llama-shaped
from superfloat import disable_tf32

REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
OUT = os.environ.get("BFCL_OUT", "/workspace/results")


def load_category(cat):
    from huggingface_hub import hf_hub_download
    q = hf_hub_download(REPO, f"BFCL_v3_{cat}.json", repo_type="dataset")
    rows = [json.loads(l) for l in open(q) if l.strip()]
    ans = {}
    if cat != "irrelevance":
        a = hf_hub_download(REPO, f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset")
        ans = {r["id"]: r["ground_truth"] for r in (json.loads(l) for l in open(a) if l.strip())}
    return rows, ans


def to_openai_tools(funcs):
    out = []
    for f in funcs:
        p = dict(f.get("parameters") or {})
        if p.get("type") == "dict":
            p["type"] = "object"                       # BFCL uses `dict`
        out.append({"type": "function", "function": {
            "name": f["name"], "description": f.get("description", ""), "parameters": p}})
    return out


CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_calls(text):
    """Qwen emits <tool_call>{...}</tool_call>. Fall back to a bare JSON object
    with name/arguments, which is what most models produce when they drift."""
    calls = []
    for m in CALL_RE.findall(text):
        try:
            d = json.loads(m)
            if isinstance(d, dict) and "name" in d:
                calls.append((d["name"], d.get("arguments") or d.get("parameters") or {}))
        except Exception:
            pass
    if not calls:
        for m in re.findall(r'\{[^{}]*"name"\s*:.*?\}(?=\s*$|\s*\n)', text, re.S):
            try:
                d = json.loads(m)
                if isinstance(d, dict) and "name" in d:
                    calls.append((d["name"], d.get("arguments") or {}))
            except Exception:
                pass
    return calls


def _val_ok(got, allowed):
    """BFCL lists every acceptable value per parameter; '' means omittable."""
    if not isinstance(allowed, list):
        allowed = [allowed]
    for a in allowed:
        if got == a:
            return True
        if isinstance(a, str) and isinstance(got, str) and got.strip().lower() == a.strip().lower():
            return True
        try:
            if isinstance(a, (int, float)) and isinstance(got, (int, float)) and float(got) == float(a):
                return True
            if isinstance(a, (int, float)) and isinstance(got, str) and float(got) == float(a):
                return True
        except Exception:
            pass
    return False


def ast_match(calls, ground_truth):
    """One call per ground-truth entry, right name, every parameter acceptable.
    A parameter may be omitted only when '' is among its allowed values."""
    if len(calls) != len(ground_truth):
        return False
    for (name, args), gt in zip(calls, ground_truth):
        gname = list(gt.keys())[0]
        if name != gname:
            return False
        spec = gt[gname]
        if not isinstance(args, dict):
            return False
        for k, allowed in spec.items():
            if k not in args:
                if isinstance(allowed, list) and "" in allowed:
                    continue
                return False
            if not _val_ok(args[k], allowed):
                return False
        for k in args:
            if k not in spec:
                return False
    return True


@torch.no_grad()
def run(model, tok, rows, answers, cat, batch, max_new, dump=None):
    hits = total = 0; called = 0; records = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i+batch]
        prompts = []
        for r in chunk:
            msgs = r["question"][0] if isinstance(r["question"][0], list) else r["question"]
            prompts.append(tok.apply_chat_template(
                msgs, tools=to_openai_tools(r["function"]),
                tokenize=False, add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=3072).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, text in zip(chunk, outs):
            calls = parse_calls(text)
            if cat == "irrelevance":
                ok = (len(calls) == 0)          # correct answer is to not call
            else:
                ok = ast_match(calls, answers.get(r["id"], []))
            hits += ok; total += 1; called += (len(calls) > 0)
            if dump is not None and len(records) < 5:
                records.append({"id": r["id"], "ok": bool(ok), "out": text[:300]})
    return hits, total, called, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--bits", type=int, default=0, help="0 = bf16 reference")
    ap.add_argument("--mode", default="ln_all", choices=["plain", "ln", "ln_all"])
    ap.add_argument("--categories", default="simple,multiple,irrelevance")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    disable_tf32()
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()

    nq = 0
    if a.bits:
        # quantize in fp32 so the grid is exact, then hand back bf16 weights,
        # which hold SF8 and coarser without loss (verified numerically)
        model = model.float()
        nq = install(model, a.bits, a.mode)
        nf, worst = fold(model)
        model = model.bfloat16()
        print(f"quantized {nq} matmuls, folded {nf}, max off-grid {worst:.2e}", flush=True)

    tag = f"bfcl_{a.model.split('/')[-1]}_" + ("bf16" if not a.bits else f"sf{a.bits}_{a.mode}")
    os.makedirs(OUT, exist_ok=True)
    res, t0 = {}, time.time()
    for cat in a.categories.split(","):
        rows, ans = load_category(cat)
        if a.limit: rows = rows[:a.limit]
        h, t, c, samples = run(model, tok, rows, ans, cat, a.batch, a.max_new, dump=True)
        res[cat] = {"correct": h, "total": t, "acc": h/t, "call_rate": c/t, "samples": samples}
        print(f"[{tag}] {cat:12s} {h:4d}/{t:4d} = {100*h/t:5.1f}%   "
              f"called on {100*c/t:5.1f}% of prompts", flush=True)
    overall = sum(v["correct"] for v in res.values()) / sum(v["total"] for v in res.values())
    rec = {"exp": "bfcl", "model": a.model, "bits": a.bits, "mode": a.mode if a.bits else "-",
           "quantized": nq, "overall": overall, "categories": res,
           "minutes": (time.time()-t0)/60, "complete": True}
    json.dump(rec, open(f"{OUT}/{tag}.json", "w"))
    print(f"[{tag}] OVERALL {100*overall:.1f}%  ({rec['minutes']:.0f}m)", flush=True)


if __name__ == "__main__":
    main()
