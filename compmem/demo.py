"""Tiny sanity tests for Phase 1 + 2 (20 dummy sentences, as the mentors suggested)."""
import random
import torch

from chunker import chunk_text
from encoder import HashEncoder, to_int
from mih import MIHIndex, MemoryStore, hamming

torch.manual_seed(0); random.seed(0)

SENTENCES = [
    "Hello, how are you?", "I am fine, thanks.", "Hi there!", "Good morning.",
    "Operating systems schedule processes using round robin.",
    "The scheduler uses a time quantum for each process.",
    "Deadlocks need mutual exclusion and circular wait.",
    "Binary search runs in logarithmic time.",
    "Quick sort has average complexity n log n.",
    "A hash table gives constant time lookups.",
    "Transformers use self attention over all tokens.",
    "Attention cost grows quadratically with length.",
    "The KV cache stores keys and values of past tokens.",
    "Gradient descent updates weights using the loss gradient.",
    "Backpropagation applies the chain rule layer by layer.",
    "SQL joins combine rows from two tables.",
    "Normalization reduces redundancy in a database.",
    "Java uses garbage collection to free memory.",
    "The exam is next Friday.", "Please remind me to revise graphs.",
]

# ---------- Phase 1 ----------
text = " ".join(SENTENCES * 20)                 # long fake conversation
chunks = chunk_text(text, 256)
print(f"[chunker] {len(chunks)} chunks; longest = {max(len(c.split()) for c in chunks)} words")

enc = HashEncoder()
enc.train()
ids = enc.batch_ids(SENTENCES)
h, h_tilde = enc(ids)
assert set(h.unique().tolist()) <= {-1.0, 1.0}, "forward output must be exactly +-1"
print("[encoder] output shape", tuple(h.shape), "values", sorted(set(h.unique().tolist())))

# gradient flow through STE
target = torch.sign(torch.randn_like(h))
loss = ((h - target) ** 2).mean()
loss.backward()
gn = sum(p.grad.norm().item() for p in enc.parameters() if p.grad is not None)
print(f"[STE] loss={loss.item():.3f}  total grad norm={gn:.4f}  (must be > 0)")
assert gn > 0

# tiny training run: make two similar sentences get closer codes
opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
a = enc.batch_ids(["The scheduler uses a time quantum for each process."])
b = enc.batch_ids(["Operating systems schedule processes using round robin."])
c = enc.batch_ids(["Hello, how are you?"])
for step in range(60):
    ha, _ = enc(a); hb, _ = enc(b); hc, _ = enc(c)
    pos = ((ha - hb) ** 2).mean(); neg = torch.relu(2.0 - ((ha - hc) ** 2).mean())
    loss = pos + neg
    opt.zero_grad(); loss.backward(); opt.step()
enc.eval()
with torch.no_grad():
    ca, cb, cc = [to_int(enc(x)[0][0]) for x in (a, b, c)]
print(f"[train] hamming(similar)={hamming(ca, cb)}  hamming(different)={hamming(ca, cc)}")

# ---------- Phase 2 ----------
# 1) MIH exactness vs brute force on random codes with planted near neighbours
idx = MIHIndex()
codes = [random.getrandbits(64) for _ in range(3000)]
for i, c_ in enumerate(codes):
    idx.add(i, c_)
for i in range(300):                            # plant neighbours within 0-7 bits
    c_ = codes[i]
    for b_ in random.sample(range(64), random.randint(0, 7)):
        c_ ^= (1 << b_)
    idx.add(10_000 + i, c_)
bad = 0
for q in codes[:300] + [random.getrandbits(64) for _ in range(100)]:
    if idx.search(q, 7) != idx.brute_force(q, 7):
        # ties may be ordered differently, compare as sets
        if set(idx.search(q, 7)) != set(idx.brute_force(q, 7)):
            bad += 1
print(f"[MIH] mismatches vs brute force over 400 queries: {bad}  (must be 0)")
assert bad == 0

# 2) clustering on real encoded dummy sentences
store = MemoryStore(cluster_radius=4, search_radius=7)
codes20 = [to_int(r) for r in enc.encode_texts(SENTENCES)]
for s, c_ in zip(SENTENCES, codes20):
    store.save(s, c_)
store.save(SENTENCES[5], codes20[5])            # exact duplicate -> must join a cluster
print(f"[cluster] {len(store.chunk_cluster)} chunks -> {len(store.clusters)} clusters")
hits = store.query(codes20[5])
print(f"[query] '{SENTENCES[5][:40]}...' -> {[(i, d) for i, d, _, _ in hits]}")
assert len(store.clusters) < len(store.chunk_cluster)
print("ALL CHECKS PASSED")
