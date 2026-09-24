"""Fold the NIAH JSONs into the tables used in the write-up.

Convention, fixed for every table in this study: **our own bf16 control is the
baseline at 100%**, and SF8 is quoted relative to it. Meta's published Llama
numbers are never the reference -- different corpus, token budget and recipe,
so any comparison to them is confounded. Directions differ by metric
(perplexity lower-is-better, retrieval higher-is-better), so each table says
which way is good.
"""
import glob, json, os, sys

NOISE = 0.05  # nats; below this a retrieval score is indistinguishable from 0


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(f))
        src = os.path.splitext(os.path.basename(r["ckpt"]))[0]
        out[(src, r["bits"], r["ctx"])] = r
    return out


def main(d="benchmarks/results/niah"):
    R = load(d)
    ctxs = sorted({k[2] for k in R})
    print("Needle retrieval, nats (higher = better). bf16 = 100%.\n")
    print(f"{'model':<16}{'ctx':>8}{'bf16':>9}{'SF8':>9}{'SF8 rel':>14}")
    for src_bf, src_sf, label in [("ext128k_bf16_model", "ext128k_sf8_model", "128K-extended"),
                                  ("bf16_model", "sf8_model", "8K base (ctrl)")]:
        for c in ctxs:
            b, s = R.get((src_bf, 0, c)), R.get((src_sf, 8, c))
            if not b or not s:
                continue
            mb, ms = b["mean_retrieval"], s["mean_retrieval"]
            rel = "n/a (noise)" if max(mb, ms) < NOISE else f"{100*ms/mb:.1f}%"
            print(f"{label:<16}{c:>8}{mb:>9.3f}{ms:>9.3f}{rel:>14}")
    print("\nBy needle distance, 128K-extended model (nats):\n")
    print(f"{'ctx':>8}{'depth':>8}{'~tokens back':>14}{'bf16':>9}{'SF8':>9}")
    for c in ctxs:
        b, s = R.get(("ext128k_bf16_model", 0, c)), R.get(("ext128k_sf8_model", 8, c))
        if not b or not s:
            continue
        for rb, rs in zip(b["rows"], s["rows"]):
            back = int((1 - rb["depth"]) * c)
            print(f"{c:>8}{rb['depth']:>8.2f}{back:>14,}{rb['retrieval']:>9.2f}{rs['retrieval']:>9.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/niah")
