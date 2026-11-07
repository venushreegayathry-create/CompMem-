"""Phase 3C on REAL multi-hop QA (HotpotQA distractor) -- trains the barcode maker + keep layer.

Pipeline per question ("episode"):
    memory = 10 HotpotQA paragraphs (2 gold + 8 distractors) + 6 small-talk chunks  -> 16 chunks
    query  = the question
Encoder  : frozen pretrained Transformer (MiniLM) -> trainable hash head -> 64 values -> Binary Concrete + STE -> +-1
Loss     : multi-positive InfoNCE on the barcode match (gold paragraphs must match the question)
           + bit-balance penalty (every bit ~50% +1, stops codes from collapsing)
Keep layer r_j = sigmoid(W mu_j + b): trained so gold chunks -> high r, noise -> low r, plus lambda*sum(r).
Outputs: compmem_hotpot.pt, results.json, training_curves.png (loss + GRADIENT NORMS for Sir's review)

Run:  python train_hotpot.py                       # real HotpotQA + MiniLM (Colab / your PC)
      python train_hotpot.py --data mock --embedder hash --n_train 800 --n_test 200 --epochs 15   # offline smoke test
"""
import argparse, json, os, random, string, urllib.request, zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mih import MemoryStore
from encoder import to_int
from llm_link import retrieve, match_score, KeepScorer

N_PARA, N_TALK, K_RET, N_BITS = 10, 6, 4, 64
DEV_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

# ------------------------------------------------------------------ data
GREET = ["Hello, how are you?", "Hi there!", "Good morning!", "Hey, what's up?", "Thanks, see you later.",
         "Nice weather today.", "I am fine, thanks.", "Okay, sounds good.", "Please remind me about the exam.",
         "Have a great day!", "Good night, talk tomorrow.", "Haha, that is funny.", "Sure, no problem.",
         "How was your weekend?", "Thank you so much!", "See you soon.", "Alright, let's continue.",
         "Sorry, I was away for a moment.", "Welcome back!", "Take care!"]


def talk_pool(n=60, seed=0):
    r = random.Random(seed)
    out = list(GREET)
    while len(out) < n:
        out.append(f"{r.choice(GREET[:8])} {r.choice(GREET[8:])}")
    return out


def _norm(item):
    """Return {question, answer, paras:[(title, text)], gold:set(indices)}."""
    if "context" in item and isinstance(item["context"], dict):          # HF datasets format
        titles, sents = item["context"]["title"], item["context"]["sentences"]
        gold_titles = set(item["supporting_facts"]["title"])
    else:                                                                # original json format
        titles = [c[0] for c in item["context"]]; sents = [c[1] for c in item["context"]]
        gold_titles = {sf[0] for sf in item["supporting_facts"]}
    paras = [(t, " ".join(s).strip()) for t, s in zip(titles, sents)]
    gold = {i for i, (t, _) in enumerate(paras) if t in gold_titles}
    return {"question": item["question"], "answer": item["answer"], "paras": paras, "gold": gold}


def make_mock(n, seed=0):
    r = random.Random(seed)
    syl = ["ka", "to", "mi", "ra", "zen", "lo", "vex", "dar", "pul", "sun", "ax", "bor", "wen", "til"]
    word = lambda: "".join(r.choice(syl) for _ in range(3))
    cities = [word() for _ in range(30)]
    out = []
    for _ in range(n):
        p, c, y = word(), word(), r.choice(cities)
        paras = [(p, f"{p} works at {c}. {p} likes music."), (c, f"{c} is located in {y}. {c} was founded long ago.")]
        for _ in range(N_PARA - 2):
            p2, c2, y2 = word(), word(), r.choice(cities)
            paras.append((p2, f"{p2} works at {c2}. {c2} is located in {y2}."))
        order = list(range(N_PARA)); r.shuffle(order)
        paras = [paras[i] for i in order]
        gold = {order.index(0), order.index(1)}
        out.append({"question": f"In which city is the company where {p} works located?", "answer": y,
                    "paras": paras, "gold": gold})
    return out


def load_hotpot(n_total, source="auto", json_path=None):
    if source == "mock":
        return make_mock(n_total)
    items = None
    if json_path:
        items = json.load(open(json_path))
    else:
        try:                                         # route 1: HuggingFace datasets
            from datasets import load_dataset
            items = list(load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"))
        except Exception as e:
            print("datasets route failed:", repr(e)[:120], "-> downloading official dev json")
            path = "hotpot_dev_distractor_v1.json"
            if not os.path.exists(path):
                urllib.request.urlretrieve(DEV_URL, path)   # route 2: official site (~47 MB)
            items = json.load(open(path))
    eps = [_norm(it) for it in items]
    eps = [e for e in eps if len(e["paras"]) >= N_PARA and len(e["gold"]) == 2]
    for e in eps:                                    # keep exactly 10 paragraphs, keep the 2 gold ones
        keep = sorted(e["gold"]) + [i for i in range(len(e["paras"])) if i not in e["gold"]][:N_PARA - 2]
        e["paras"] = [e["paras"][i] for i in keep]; e["gold"] = {0, 1}
        perm = list(range(N_PARA)); random.Random(zlib.crc32(e["question"].encode())).shuffle(perm)
        e["paras"] = [e["paras"][i] for i in perm]; e["gold"] = {perm.index(0), perm.index(1)}
    random.Random(0).shuffle(eps)
    return eps[:n_total]


# ------------------------------------------------------------------ embedders (frozen pretrained Transformer)
class Embedder:
    def __init__(self, name="minilm", dim=384, device="cpu"):
        self.name, self.dim, self.device = name, dim, device
        if name == "minilm":
            from sentence_transformers import SentenceTransformer
            self.m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
            self.m.max_seq_length = 256               # = chunk size from the SOW
            self.dim = self.m.get_sentence_embedding_dimension()
        else:                                          # offline fallback: random-projection bag of words
            g = torch.Generator().manual_seed(0)
            self.proj = torch.randn(65536, dim, generator=g)

    @torch.no_grad()
    def __call__(self, texts, bs=128):
        if self.name == "minilm":
            return self.m.encode(texts, batch_size=bs, convert_to_tensor=True, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 500).cpu().float()
        out = []
        for t in texts:
            ids = [zlib.crc32(w.encode()) % 65536 for w in t.lower().replace(".", " ").replace("?", " ").split()]
            v = self.proj[ids].mean(0) if ids else torch.zeros(self.dim)
            out.append(F.normalize(v, dim=0))
        return torch.stack(out)


def para_text(t, s):
    return f"{t}: {s}"


def embed_dataset(eps, emb, talk):
    """-> Q (n,d), P (n,10,d), T (len(talk),d)"""
    Q = emb([e["question"] for e in eps])
    P = emb([para_text(t, s) for e in eps for t, s in e["paras"]]).view(len(eps), N_PARA, -1)
    T = emb(talk)
    return Q, P, T


# ------------------------------------------------------------------ hash head (Phase 1 B+C on top of the encoder)
class HashHead(nn.Module):
    def __init__(self, d_in, n_bits=N_BITS, hidden=512, tau=1.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, n_bits))
        self.tau = tau

    def forward(self, x):
        z = self.net(x)
        if self.training:                                          # Binary Concrete: logistic noise + tanh
            u = torch.rand_like(z).clamp(1e-6, 1 - 1e-6)
            h_tilde = torch.tanh((z + torch.log(u) - torch.log(1 - u)) / self.tau)
        else:
            h_tilde = torch.tanh(z / self.tau)
        sign = torch.where(h_tilde >= 0, 1.0, -1.0).to(h_tilde.dtype)
        h = sign + (h_tilde - h_tilde.detach())                    # straight-through estimator
        return h, h_tilde


def infonce(s, gold, temp=0.1):
    """s (B,n) barcode similarities in [-1,1]; gold (B,n) bool. Multi-positive InfoNCE."""
    logits = s / temp
    pos = torch.logsumexp(logits.masked_fill(~gold, -1e9), dim=1)
    return (torch.logsumexp(logits, dim=1) - pos).mean()


def make_batch(Q, P, T, gold, idx, g):
    B = len(idx)
    talk_idx = torch.randint(0, T.size(0), (B, N_TALK), generator=g)
    M = torch.cat([P[idx], T[talk_idx]], 1)                                    # (B,16,d)
    lab = torch.cat([gold[idx], torch.zeros(B, N_TALK, dtype=torch.bool)], 1)  # gold mask
    return Q[idx], M, lab


def gold_tensor(eps):
    g = torch.zeros(len(eps), N_PARA, dtype=torch.bool)
    for i, e in enumerate(eps):
        for j in e["gold"]:
            g[i, j] = True
    return g


@torch.no_grad()
def quick_recall(head, Q, P, T, gold, k=K_RET, seed=1):
    """Brute-force Hamming recall@k: both gold paragraphs in top-k."""
    head.eval(); g = torch.Generator().manual_seed(seed)
    idx = torch.arange(len(Q))
    q, M, lab = make_batch(Q, P, T, gold, idx, g)
    hq, _ = head(q); hm, _ = head(M)
    ham = (N_BITS - (hm * hq.unsqueeze(1)).sum(-1)) / 2 + torch.rand(hm.shape[:2]) * 1e-3   # random tie-break
    top = ham.topk(k, dim=1, largest=False).indices
    got = torch.gather(lab, 1, top).sum(1)
    return (got == 2).float().mean().item(), (got.float() / 2).mean().item()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="auto", help="auto | mock")
    ap.add_argument("--json", default=None, help="path to hotpot_dev_distractor_v1.json (if download fails)")
    ap.add_argument("--embedder", default="minilm", help="minilm | hash")
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lam", type=float, default=0.002, help="penalty lambda * sum(r)")
    ap.add_argument("--balance", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0); random.seed(0)

    eps = load_hotpot(a.n_train + a.n_test, a.data, a.json)
    train, test = eps[:a.n_train], eps[a.n_train:a.n_train + a.n_test]
    print(f"episodes: train {len(train)}  test {len(test)}")
    cache = f"emb_{a.embedder}_{a.data}_{a.n_train}_{a.n_test}.pt"
    talk = talk_pool()
    if os.path.exists(cache):
        Qtr, Ptr, Qte, Pte, T = torch.load(cache)
    else:
        emb = Embedder(a.embedder, device=a.device)
        Qtr, Ptr, T = embed_dataset(train, emb, talk)
        Qte, Pte, _ = embed_dataset(test, emb, talk)
        torch.save((Qtr, Ptr, Qte, Pte, T), cache)
    Gtr, Gte = gold_tensor(train), gold_tensor(test)
    d = Qtr.size(1)

    head = HashHead(d)
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    g = torch.Generator().manual_seed(0)
    hist = {"loss": [], "grad": [], "recall": [], "ep_recall": []}
    base_both = (K_RET / (N_PARA + N_TALK)) * ((K_RET - 1) / (N_PARA + N_TALK - 1))
    print(f"random baseline: both gold in top-{K_RET} = {base_both:.3f}")
    for ep in range(a.epochs):
        head.train(); perm = torch.randperm(len(Qtr)); tot = 0
        for b in range(0, len(perm), a.bs):
            idx = perm[b:b + a.bs]
            q, M, lab = make_batch(Qtr, Ptr, T, Gtr, idx, g)
            hq, htq = head(q); hm, _ = head(M)
            s = (hm * htq.unsqueeze(1)).sum(-1) / N_BITS                       # alpha-like barcode match
            bal = (hm.reshape(-1, N_BITS).mean(0) ** 2).mean() + (htq.mean(0) ** 2).mean()
            loss = infonce(s, lab) + a.balance * bal
            opt.zero_grad(); loss.backward()
            gn = torch.sqrt(sum((p.grad ** 2).sum() for p in head.parameters())).item()
            opt.step()
            hist["loss"].append(loss.item()); hist["grad"].append(gn); tot += loss.item() * len(idx)
        both, frac = quick_recall(head, Qte, Pte, T, Gte)
        hist["recall"].append(both); hist["ep_recall"].append(ep + 1)
        print(f"epoch {ep+1:3d}  loss {tot/len(perm):.3f}  grad-norm {np.mean(hist['grad'][-10:]):.3f}  "
              f"test recall@{K_RET} (both gold) {both:.3f}  (random {base_both:.3f})")

    # ---------------- final retrieval through the real MIH store
    head.eval()
    with torch.no_grad():
        hq_all, _ = head(Qte); hP, _ = head(Pte); hT, _ = head(T)
    rg = random.Random(5); rec_mih = 0; frac_mih = 0; bit_pm = hP.reshape(-1, N_BITS).mean(0)
    keep_rows = []                                      # (mu, label) for keep-layer training later
    for i in range(len(test)):
        tk = rg.sample(range(len(talk)), N_TALK)
        codes = [to_int(h) for h in hP[i]] + [to_int(hT[j]) for j in tk]
        store = MemoryStore()
        for n, c in enumerate(codes):
            store.save(f"chunk{n}", c)
        ids = retrieve(store, to_int(hq_all[i]), k=K_RET)
        got = len(set(ids) & test[i]["gold"])
        rec_mih += got == 2; frac_mih += got / 2
    rec_mih /= len(test); frac_mih /= len(test)
    print(f"\nTEST via MIH store: both-gold recall@{K_RET} = {rec_mih:.3f} | per-gold recall = {frac_mih:.3f} "
          f"| random baseline both = {base_both:.3f}")
    print(f"bit balance: mean |avg bit| = {bit_pm.abs().mean():.3f} (0 = perfectly balanced)")

    # ---------------- keep layer r_j = sigmoid(W mu_j + b)  (cluster-average barcodes)
    def keep_data(Qs, Ps, Gs, hP_, hT_, n, seed):
        r = random.Random(seed); X, Y, KIND = [], [], []
        for i in range(n):
            tk = r.sample(range(len(talk)), N_TALK)
            hs = list(hP_[i]) + [hT_[j] for j in tk]
            store = MemoryStore()
            for n_, h in enumerate(hs):
                store.save(f"c{n_}", to_int(h))
            for n_ in range(len(hs)):
                X.append(torch.tensor(store.clusters[store.chunk_cluster[n_]].mu))
                Y.append(1.0 if (n_ < N_PARA and Gs[i, n_]) else 0.0)
                KIND.append("gold" if Y[-1] else ("distractor" if n_ < N_PARA else "smalltalk"))
        return torch.stack(X), torch.tensor(Y), KIND
    with torch.no_grad():
        hPtr, _ = head(Ptr); hTt, _ = head(T)
    nk = min(len(train), 1500)
    Xtr, Ytr, _ = keep_data(Qtr, Ptr, Gtr, hPtr, hTt, nk, 11)
    Xte, Yte, Kte = keep_data(Qte, Pte, Gte, hP, hT, len(test), 12)
    keep = KeepScorer()
    kopt = torch.optim.Adam(keep.parameters(), lr=1e-2)
    for step in range(400):
        r_ = keep(Xtr)
        kl = F.binary_cross_entropy(r_, Ytr) + a.lam * r_.sum() / len(Xtr) * N_PARA   # + lambda * sum(r)
        kopt.zero_grad(); kl.backward(); kopt.step()
    with torch.no_grad():
        r_te = keep(Xte)
    kinds = np.array(Kte); rmean = {k: float(r_te[torch.tensor(kinds == k)].mean()) for k in ("gold", "distractor", "smalltalk")}
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(Yte.numpy(), r_te.numpy()))
    print(f"keep score r (test): gold {rmean['gold']:.3f} | distractor {rmean['distractor']:.3f} | "
          f"small-talk {rmean['smalltalk']:.3f} | AUC gold-vs-rest {auc:.3f}")

    torch.save({"head": head.state_dict(), "keep": keep.state_dict(), "d_in": d, "embedder": a.embedder},
               "compmem_hotpot.pt")
    json.dump({"random_baseline_both": base_both, "test_recall_both_MIH": rec_mih, "test_recall_per_gold_MIH": frac_mih,
               "final_epoch_recall_bruteforce": hist["recall"][-1], "keep_r_mean": rmean, "keep_auc": auc,
               "bit_imbalance": float(bit_pm.abs().mean()), "n_train": len(train), "n_test": len(test),
               "embedder": a.embedder}, open("results.json", "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(hist["loss"]); ax[0].set(title="Training loss", xlabel="step")
    ax[1].plot(hist["grad"]); ax[1].set(title="Gradient norm through the STE (must stay > 0)", xlabel="step", yscale="log")
    ax[2].plot(hist["ep_recall"], hist["recall"], "g-o", label="CompMem hash codes")
    ax[2].axhline(base_both, color="r", ls="--", label="random baseline")
    ax[2].set(title=f"Test recall@{K_RET} (both gold paragraphs)", xlabel="epoch", ylim=(0, 1)); ax[2].legend()
    plt.tight_layout(); plt.savefig("training_curves.png", dpi=150)
    print("saved compmem_hotpot.pt, results.json, training_curves.png")


if __name__ == "__main__":
    main()
