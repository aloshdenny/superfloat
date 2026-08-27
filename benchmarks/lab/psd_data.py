"""Corpora for the pure-SF datapath study, tokenized with SmolLM2's tokenizer.

Same two corpora as smol_qat.py -- general prose and tool-call traces -- with one
change that matters for a study run in pieces over days. smol_qat.py takes the
last 2% of each file as validation, so the held-out text moves whenever the file
length changes and numbers from a 30M-token prepare cannot be compared against a
250M-token one. Here the *first* VAL_TOKENS of each corpus are the validation
region and everything after is training data, so the held-out text is fixed the
moment the first token is written and every later run measures the same thing.

Stored as uint16, which SmolLM2's 49152-entry vocabulary fits with room to
spare; smol_qat.py used uint32 and paid twice the page cache for it.

    python psd_data.py --general 150000000
"""
import argparse, os
import numpy as np

MODEL = "HuggingFaceTB/SmolLM2-360M"
DATA = os.environ.get("PSD_DATA", os.path.expanduser("~/sf-psd/data"))
VAL_TOKENS = 4_000_000
VAL_TOKENS_TOOL = 2_000_000


def _render_glaive(r):
    return (r.get("system") or "").strip() + "\n" + (r.get("chat") or "").strip()


def _render_hermes(r):
    out = [f"SYSTEM: You have access to the following tools:\n{r.get('tools') or ''}"]
    for m in (r.get("conversations") or []):
        out.append(f"{(m.get('from') or '').upper()}: {m.get('value','')}")
    return "\n".join(out)


def prepare(n_general, out_dir=DATA):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    os.makedirs(out_dir, exist_ok=True)
    tk = AutoTokenizer.from_pretrained(MODEL)
    if tk.vocab_size >= 2 ** 16:
        raise RuntimeError(f"vocab {tk.vocab_size} does not fit uint16")
    eos = tk.eos_token_id

    g = os.path.join(out_dir, "general.bin")
    if not (os.path.exists(g) and os.path.getsize(g) == n_general * 2):
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        buf = np.memmap(g, dtype=np.uint16, mode="w+", shape=(n_general,))
        i, batch = 0, []

        def flush(b, i):
            for ids in tk(b)["input_ids"]:
                ids = ids + [eos]
                if i + len(ids) > n_general:
                    ids = ids[:n_general - i]
                if not ids:
                    break
                buf[i:i + len(ids)] = np.array(ids, dtype=np.uint16)
                i += len(ids)
            return i

        for row in ds:
            batch.append(row["text"])
            if len(batch) >= 2000:
                i = flush(batch, i)
                batch = []
                print(f"  general {i/1e6:.0f}M / {n_general/1e6:.0f}M", flush=True)
                if i >= n_general:
                    break
        if i < n_general:
            i = flush(batch, i)
        buf.flush(); del buf
        print(f"general: {i/1e6:.0f}M tokens -> {g}", flush=True)

    t = os.path.join(out_dir, "tool.bin")
    if not os.path.exists(t):
        texts = []
        try:
            d = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
            texts += [_render_glaive(r) for r in d]
            print(f"glaive {len(d)}", flush=True)
        except Exception as e:
            print("glaive:", str(e)[:100], flush=True)
        for cfg in ("func_calling_singleturn", "func_calling", "glaive_func_calling"):
            try:
                d = load_dataset("NousResearch/hermes-function-calling-v1", cfg,
                                 split="train")
                texts += [_render_hermes(r) for r in d]
                print(f"hermes/{cfg} {len(d)}", flush=True)
            except Exception as e:
                print(f"hermes/{cfg}:", str(e)[:80], flush=True)
        if not texts:
            raise RuntimeError("no tool corpus available")
        ids = []
        for j in range(0, len(texts), 1000):
            for e in tk(texts[j:j + 1000])["input_ids"]:
                ids.extend(e)
                ids.append(eos)
        arr = np.array(ids, dtype=np.uint16)
        np.memmap(t, dtype=np.uint16, mode="w+", shape=arr.shape)[:] = arr
        print(f"tool: {len(arr)/1e6:.1f}M tokens from {len(texts)} traces", flush=True)


class Mix:
    """Batches drawn entirely from one corpus, so a sequence is never half prose
    and half JSON. Validation batches are drawn by a tag-seeded generator, so
    every configuration in a sweep sees byte-identical text -- the section 8
    caveat on smol_qat.py's PTQ sweep, which let the batches drift between
    configurations and cost it any resolution below 0.02 nats."""

    def __init__(self, d=DATA, seqlen=1024, p_tool=0.25, seed=0, device="cpu"):
        self.g = np.memmap(os.path.join(d, "general.bin"), dtype=np.uint16, mode="r")
        self.t = np.memmap(os.path.join(d, "tool.bin"), dtype=np.uint16, mode="r")
        self.gv, self.tv = VAL_TOKENS, VAL_TOKENS_TOOL
        if len(self.g) <= self.gv or len(self.t) <= self.tv:
            raise RuntimeError("corpus smaller than its validation region")
        self.L, self.p, self.dev = seqlen, p_tool, device
        self.rng = np.random.default_rng(seed)

    def _take(self, src, lo, hi, n, rng):
        import torch
        ix = rng.integers(lo, hi - self.L - 1, size=n)
        x = np.stack([src[i:i + self.L] for i in ix]).astype(np.int64)
        y = np.stack([src[i + 1:i + 1 + self.L] for i in ix]).astype(np.int64)
        return (torch.from_numpy(x).to(self.dev), torch.from_numpy(y).to(self.dev))

    def train(self, n):
        if self.rng.random() < self.p:
            return self._take(self.t, self.tv, len(self.t), n, self.rng)
        return self._take(self.g, self.gv, len(self.g), n, self.rng)

    def _fixed(self, src, hi, n, tag, batch_index):
        # hashlib, not hash(): PYTHONHASHSEED randomises the builtin across
        # processes, which would silently change the held-out batches.
        import hashlib
        seed = int.from_bytes(hashlib.md5(f"{tag}:{batch_index}".encode()).digest()[:8], "little") % (2 ** 31)
        rng = np.random.default_rng(seed)
        return self._take(src, 0, hi, n, rng)

    def val_general(self, n, i=0):
        return self._fixed(self.g, self.gv, n, "gen", i)

    def val_tool(self, n, i=0):
        return self._fixed(self.t, self.tv, n, "tool", i)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--general", type=int, default=150_000_000)
    ap.add_argument("--out", default=DATA)
    a = ap.parse_args()
    prepare(a.general, a.out)
