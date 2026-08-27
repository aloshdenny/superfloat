"""C → C++ translation capability under SF PTQ.

Public proxy for the fighter-jet / avionics translation claim: short C
functions must come back as C++ that `clang++ -fsyntax-only` accepts.
Scored as compile-pass rate, not BLEU. Same Qwen2.5-Coder + ln_all recipe
as the tool-use eval.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smol_qat import install, fold
from superfloat import disable_tf32
from bfcl_eval import pick_device

_HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("DOMAIN_OUT", _HERE.parent / "results" / "domain"))

SNIPPETS = [
    "int add(int a, int b) { return a + b; }",
    "int abs_i(int x) { return x < 0 ? -x : x; }",
    "int fact(int n) { int p = 1; for (int i = 1; i <= n; i++) p *= i; return p; }",
    "int max3(int a, int b, int c) { int m = a > b ? a : b; return m > c ? m : c; }",
    "unsigned bitcount(unsigned x) { unsigned n = 0; while (x) { n += x & 1u; x >>= 1; } return n; }",
    "int clamp(int x, int lo, int hi) { if (x < lo) return lo; if (x > hi) return hi; return x; }",
    "double lerp(double a, double b, double t) { return a + (b - a) * t; }",
    "int gcd(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a < 0 ? -a : a; }",
    "void swap_i(int *a, int *b) { int t = *a; *a = *b; *b = t; }",
    "int sum_n(const int *a, int n) { int s = 0; for (int i = 0; i < n; i++) s += a[i]; return s; }",
    "int saturating_add(int a, int b) { long long c = (long long)a + b; if (c > 2147483647LL) return 2147483647; if (c < -2147483648LL) return (int)(-2147483648LL); return (int)c; }",
    "float clampf(float x, float lo, float hi) { if (x < lo) return lo; if (x > hi) return hi; return x; }",
    "int is_pow2(unsigned x) { return x && !(x & (x - 1)); }",
    "unsigned rotl(unsigned x, int k) { k &= 31; return (x << k) | (x >> (32 - k)); }",
    "int sign(int x) { return (x > 0) - (x < 0); }",
    "double hypot2(double x, double y) { return x * x + y * y; }",
    "int count_eq(const int *a, int n, int v) { int c = 0; for (int i = 0; i < n; i++) if (a[i] == v) c++; return c; }",
    "void zero(int *a, int n) { for (int i = 0; i < n; i++) a[i] = 0; }",
    "int min_idx(const int *a, int n) { int m = 0; for (int i = 1; i < n; i++) if (a[i] < a[m]) m = i; return m; }",
    "char lower(char c) { return (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c; }",
    "int is_digit(char c) { return c >= '0' && c <= '9'; }",
    "unsigned next_pow2(unsigned x) { if (x <= 1) return 1; x--; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; return x + 1; }",
    "int copysign_i(int mag, int sgn) { int a = mag < 0 ? -mag : mag; return sgn < 0 ? -a : a; }",
    "void reverse(int *a, int n) { for (int i = 0, j = n - 1; i < j; i++, j--) { int t = a[i]; a[i] = a[j]; a[j] = t; } }",
    "int starts_with(const char *s, const char *p) { while (*p) { if (*s++ != *p++) return 0; } return 1; }",
    "int wrap(int x, int n) { int r = x % n; return r < 0 ? r + n : r; }",
    "double deg2rad(double d) { return d * 3.14159265358979323846 / 180.0; }",
    "int median3(int a, int b, int c) { if (a > b) { int t = a; a = b; b = t; } if (b > c) { int t = b; b = c; c = t; } if (a > b) { int t = a; a = b; b = t; } return b; }",
    "unsigned pop_lsb(unsigned x) { return x & (unsigned)(-(int)x); }",
    "int within(int x, int lo, int hi) { return x >= lo && x <= hi; }",
]


PROMPT = (
    "Translate this C function into equivalent modern C++. "
    "Output ONLY the C++ function (no markdown, no explanation). "
    "It must compile with clang++ -std=c++17 -fsyntax-only.\n\n"
    "```c\n{src}\n```\n"
)


def strip_fence(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if "```" in t:
            t = t.rsplit("```", 1)[0]
    return t.strip()


def compiles(src):
    cxx = os.environ.get("CXX", "clang++")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.cpp"
        p.write_text(src + "\n")
        r = subprocess.run([cxx, "-std=c++17", "-fsyntax-only", str(p)],
                           capture_output=True, text=True)
        return r.returncode == 0, (r.stderr or "")[-400:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--bits", type=int, default=0)
    ap.add_argument("--mode", default="ln_all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="", help="cuda | mps | cpu (default: auto)")
    a = ap.parse_args()
    disable_tf32()
    device = torch.device(a.device) if a.device else pick_device()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    nq = 0
    if a.bits:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                a.model, torch_dtype=torch.float32).eval()
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
        nq = install(model, a.bits, a.mode)
        nf, worst = fold(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = model.bfloat16().to(device)
        print(f"quantized {nq} folded {nf} offgrid {worst:.2e}", flush=True)
    else:
        dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(
                a.model, torch_dtype=dtype).to(device).eval()
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                a.model, dtype=dtype).to(device).eval()

    rows = SNIPPETS[:a.limit] if a.limit else SNIPPETS
    hits = 0
    dump = []
    t0 = time.time()
    for src in rows:
        msgs = [{"role": "user", "content": PROMPT.format(src=src)}]
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        cpp = strip_fence(text)
        ok, err = compiles(cpp)
        hits += int(ok)
        if len(dump) < 5:
            dump.append({"src": src, "ok": ok, "cpp": cpp[:400], "err": err[:200]})
        print(f"{'OK' if ok else 'FAIL':4s}  {src[:60]}", flush=True)

    rec = dict(exp="code_xlat", model=a.model, bits=a.bits, mode=a.mode if a.bits else "-",
               quantized=nq, n=len(rows), pass_rate=hits / max(len(rows), 1),
               hits=hits, minutes=(time.time() - t0) / 60, samples=dump)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"xlat_{a.model.split('/')[-1]}_" + ("bf16" if not a.bits else f"sf{a.bits}")
    json.dump(rec, open(OUT / f"{tag}.json", "w"), indent=2)
    print(f"[{tag}] {hits}/{len(rows)} = {100*rec['pass_rate']:.1f}%  {rec['minutes']:.1f}m",
          flush=True)


if __name__ == "__main__":
    main()
