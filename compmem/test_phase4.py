"""Test the cleaner: 300 noise chunks + 40 crowded useful clusters (8 chunks each) = 620 > 512."""
import asyncio, random
from mih import MemoryStore
from cleaner import maintain

random.seed(1)
store = MemoryStore()
for i in range(300):                                   # noise: random codes -> own clusters
    store.save(f"NOISE small talk {i}", random.getrandbits(64))
for g in range(40):                                    # useful but repetitive clusters
    base = random.getrandbits(64)
    for j in range(8):
        c = base ^ (1 << random.randrange(64)) if j else base   # <=1 bit from base -> same cluster
        store.save(f"fact {g}. repeated note {j}.", c)

print("before:", len(store.index.codes), "chunks,", len(store.clusters), "clusters")
keep_fn = lambda cl: 0.05 if store.texts[cl.medoid_id].startswith("NOISE") else 0.9
hash_fn = lambda text: random.getrandbits(64)
deleted, summ = asyncio.run(maintain(store, keep_fn, hash_fn=hash_fn))
print(f"deleted {deleted} noise clusters, summarized {summ} crowded clusters")
print("after: ", len(store.index.codes), "chunks,", len(store.clusters), "clusters")
assert len(store.index.codes) <= 512
assert not any(t.startswith("NOISE") for t in store.texts.values())
assert summ == 40 and deleted == 300
print("PHASE 4 CLEANER OK")
