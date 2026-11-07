"""Phase 3, Steps A + B: relevance scores and Soft KV-Modulation.

- match_score  alpha_i : how well query's smooth barcode h~_q matches memory's binary barcode h_i
- KeepScorer   r_j     : sigmoid(W_e . mu_j + b_e) -> is cluster B_j useful or noise?
- ModulatedReader      : a small attention reader. The bias log(alpha_i * r_j + 1e-6) is ADDED to the
                         attention scores of every token that came from retrieved chunk i.
                         (Same mechanism you would add inside a real LLM's attention layers.)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from chunker import token_ids


def match_score(h_tilde_q: torch.Tensor, h_mem: torch.Tensor) -> torch.Tensor:
    """h_tilde_q: (64,) in [-1,1] (smooth);  h_mem: (M,64) in {-1,+1}.  Returns (M,) in [0,1]."""
    return (1.0 + (h_mem @ h_tilde_q) / h_tilde_q.numel()) / 2.0


class KeepScorer(nn.Module):
    """r_j = sigmoid(W_e mu_j + b_e),  mu_j = average barcode of cluster j (in [-1,1]^64)."""
    def __init__(self, n_bits: int = 64):
        super().__init__()
        self.lin = nn.Linear(n_bits, 1)
        nn.init.constant_(self.lin.bias, 1.0)       # start with r ~ 0.73 so nothing is dropped at the start

    def forward(self, mu: torch.Tensor) -> torch.Tensor:    # (M,64) -> (M,)
        return torch.sigmoid(self.lin(mu)).squeeze(-1)


def retrieve(store, q_code: int, k: int = 4, r: int = 7):
    """Top-k chunk ids by Hamming distance. Uses the MIH index first (r<=7);
    if it returns fewer than k, fills up with the nearest remaining chunks (brute force)
    so that early in training (when codes are poor) the reader still gets k chunks."""
    hits = [i for i, _ in store.index.search(q_code, r)][:k]
    if len(hits) < k:
        rest = sorted((bin(q_code ^ c).count("1"), i) for i, c in store.index.codes.items()
                      if i not in hits)
        hits += [i for _, i in rest[:k - len(hits)]]
    return hits


class ModulatedReader(nn.Module):
    """Tiny 2-hop attention reader over the tokens of the retrieved chunks.
    scores = q.k / sqrt(d)  +  log(alpha_i * r_j + 1e-6)     <-- Soft KV-Modulation"""
    def __init__(self, n_answers: int, vocab: int = 8192, d: int = 64, hops: int = 2):
        super().__init__()
        self.d, self.hops = d, hops
        self.emb = nn.Embedding(vocab, d)
        self.wq, self.wk, self.wv = (nn.Linear(d, d) for _ in range(3))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_answers)

    def forward(self, q_ids, mem_ids, mem_bias):
        # q_ids (Lq,)  mem_ids (T,)  mem_bias (T,)
        q = self.emb(q_ids).mean(0)
        e = self.emb(mem_ids)
        K, V = self.wk(e), self.wv(e)
        for _ in range(self.hops):
            scores = (K @ self.wq(q)) / math.sqrt(self.d) + mem_bias
            ctx = F.softmax(scores, dim=0) @ V
            q = self.norm(q + ctx)
        return self.out(q)


def build_memory_inputs(store, chunk_ids, h_tilde_q, h_of, mu_of, keep, vocab=8192):
    """Concatenate retrieved chunk tokens and build the per-token bias log(alpha*r + 1e-6)."""
    h_mem = torch.stack([h_of[i] for i in chunk_ids])                  # (k,64) +-1
    mu = torch.stack([mu_of(store, i) for i in chunk_ids])             # (k,64)
    alpha = match_score(h_tilde_q, h_mem)                              # (k,)
    r = keep(mu)                                                       # (k,)
    bias_per_chunk = torch.log(alpha * r + 1e-6)
    ids, bias = [], []
    for n, i in enumerate(chunk_ids):
        t = token_ids(store.texts[i], vocab) or [1]
        ids += t
        bias += [bias_per_chunk[n]] * len(t)
    return torch.tensor(ids), torch.stack(bias), r
