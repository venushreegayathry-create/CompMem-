"""Phase 4, Step A: background memory cleaner (Evict vs Summarize).

Triggers when total chunks > M_MAX (512):
  Rule 1  cluster keep-score r_j < 0.25            -> delete its text + barcodes completely
  Rule 2  r_j >= 0.25 and cluster has > K_MAX (5)   -> LLM summarizes the chunks into ONE paragraph,
                                                       the summary is hashed + saved, old copies deleted
`keep_fn(cluster) -> float`   : e.g. lambda c: keep(torch.tensor(c.mu)).item()
`summarize_fn(list[str]) -> str` : plug in your LLM call (stub below just joins first sentences)
`hash_fn(str) -> int`         : text -> 64-bit code, e.g. to_int(enc.encode_texts([t])[0])
"""
import asyncio
from typing import Callable, List

M_MAX, K_MAX, KEEP_THRESHOLD = 512, 5, 0.25


def stub_summarize(texts: List[str]) -> str:
    """Placeholder for the LLM: keeps the first sentence of each chunk, de-duplicated."""
    seen, out = set(), []
    for t in texts:
        s = t.split(".")[0].strip()
        if s and s not in seen:
            seen.add(s); out.append(s)
    return ". ".join(out[:5]) + "."


def delete_cluster(store, cluster_id: int):
    cl = store.clusters.pop(cluster_id)
    for cid in cl.members:
        if cid in store.index.codes:
            store.index.remove(cid)
        store.texts.pop(cid, None)
        store.chunk_cluster.pop(cid, None)
    store.medoid_index.remove(cluster_id)


async def maintain(store, keep_fn: Callable, summarize_fn: Callable = stub_summarize,
                   hash_fn: Callable = None, m_max: int = M_MAX, k_max: int = K_MAX):
    """Async cleaner. Returns (n_deleted_clusters, n_summarized_clusters)."""
    if len(store.index.codes) <= m_max:
        return 0, 0
    deleted = summarized = 0
    for cid, cl in list(store.clusters.items()):
        score = keep_fn(cl)
        if score < KEEP_THRESHOLD:                              # Rule 1: noise
            delete_cluster(store, cid); deleted += 1
        elif len(cl.members) > k_max:                           # Rule 2: crowded -> summarize
            texts = [store.texts[m] for m in cl.members]
            summary = await asyncio.to_thread(summarize_fn, texts)   # LLM call off the main loop
            delete_cluster(store, cid)
            store.save(summary, hash_fn(summary)); summarized += 1
        await asyncio.sleep(0)                                  # yield: don't block the chatbot
    return deleted, summarized


def start_background(store, keep_fn, summarize_fn=stub_summarize, hash_fn=None, every_s=5.0):
    """Run inside an asyncio app: periodically calls maintain() as a background task."""
    async def loop():
        while True:
            await maintain(store, keep_fn, summarize_fn, hash_fn)
            await asyncio.sleep(every_s)
    return asyncio.create_task(loop())
