"""Phase 4, Step B: RAM + speed of CompMem vs a standard full-attention LLM.

Standard LLM : KV-cache grows linearly (bytes), prefill attention time grows ~ N^2.
               Time is MEASURED up to 8k tokens, then EXTRAPOLATED with a quadratic fit (dashed).
               KV memory is computed from a GPT-2-small-like config (12 layers, 12 heads x 64, fp16).
CompMem      : MEASURED for the whole range: chunk, hash, insert, MIH-retrieve, attend over k=4 chunks.
               Memory measured with tracemalloc (barcodes + hash tables + chunk text); bounded by the cleaner.
"""
import random, time, tracemalloc
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mih import MemoryStore
from cleaner import maintain, M_MAX

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=24)      # Qwen2.5-0.5B
ap.add_argument("--heads", type=int, default=14)
ap.add_argument("--kv_heads", type=int, default=2)
ap.add_argument("--dh", type=int, default=64)
ap.add_argument("--max_measured", type=int, default=0, help="0 = auto (32000 on GPU, 8000 on CPU)")
ap.add_argument("--max_n", type=int, default=100000)
ARGS = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS, HEADS, DH, BYTES, CHUNK, K = ARGS.layers, ARGS.heads, ARGS.dh, 2, 256, 4
KV_BYTES_PER_TOKEN = 2 * LAYERS * ARGS.kv_heads * DH * BYTES     # keys + values (GQA)
WORDS = [f"w{i}" for i in range(500)]


def std_attention_time(n):
    dt = torch.float16 if DEV == "cuda" else torch.float32
    q, k, v = (torch.randn(1, HEADS, n, DH, device=DEV, dtype=dt) for _ in range(3))
    if DEV == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    if DEV == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t) * LAYERS                     # all layers


def compmem_run(n_tokens):
    random.seed(0)
    tracemalloc.start()
    store = MemoryStore()
    dt = torch.float16 if DEV == "cuda" else torch.float32
    ctx_k, ctx_v = (torch.randn(1, HEADS, K * CHUNK, DH, device=DEV, dtype=dt) for _ in range(2))
    t = time.perf_counter()
    for c in range(n_tokens // CHUNK):
        text = " ".join(random.choices(WORDS, k=CHUNK))          # a 256-token chunk
        code = random.getrandbits(64)                            # stand-in for encoder output (cheap)
        store.save(text, code)
        store.query(code, 7)                                     # MIH retrieval
        q = torch.randn(1, HEADS, CHUNK, DH, device=DEV, dtype=dt)   # attend over retrieved k chunks only
        for _ in range(LAYERS):
            torch.nn.functional.scaled_dot_product_attention(q, ctx_k, ctx_v)
        if len(store.index.codes) > M_MAX:                       # keep memory bounded (evict oldest)
            oldest = min(store.texts)
            store.index.remove(oldest); store.texts.pop(oldest); store.chunk_cluster.pop(oldest)
    elapsed = time.perf_counter() - t
    mem = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    return elapsed, mem


def main():
    Ns = [n for n in [1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000] if n <= ARGS.max_n]
    lim = ARGS.max_measured or (32000 if DEV == "cuda" else 8000)
    meas_N = [n for n in Ns if n <= lim]
    std_t = [min(std_attention_time(n) for _ in range(3)) for n in meas_N]
    coef = np.polyfit(meas_N, std_t, 2)
    std_time = [float(np.polyval(coef, n)) for n in Ns]
    std_mem = [n * KV_BYTES_PER_TOKEN / 1e6 for n in Ns]                     # MB
    cm = [compmem_run(n) for n in Ns]
    cm_time, cm_mem = [c[0] for c in cm], [c[1] / 1e6 for c in cm]

    for n, a, b, c, d in zip(Ns, std_time, cm_time, std_mem, cm_mem):
        print(f"N={n:>7}  std time {a:8.2f}s  CompMem {b:6.2f}s | std KV {c:9.1f} MB  CompMem {d:6.2f} MB")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(Ns, std_mem, "r-o", label="Standard LLM (KV-cache)")
    ax[0].plot(Ns, cm_mem, "g-o", label="CompMem (barcodes+index)")
    ax[0].set(xlabel="conversation length (tokens)", ylabel="memory (MB)", title="RAM usage"); ax[0].legend()
    ax[1].plot(meas_N, std_t, "r-o", label="Standard (measured)")
    ax[1].plot(Ns, std_time, "r--", label="Standard (quadratic extrapolation beyond measured range)")
    ax[1].plot(Ns, cm_time, "g-o", label="CompMem (measured)")
    ax[1].set(xlabel="conversation length (tokens)", ylabel="time (s)", title="Time to process conversation")
    ax[1].legend()
    plt.tight_layout(); plt.savefig("benchmark.png", dpi=150)
    print("saved benchmark.png")


if __name__ == "__main__":
    main()
