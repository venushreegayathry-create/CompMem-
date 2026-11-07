"""END-TO-END training (SOW Phase 3C): the LLM's ANSWER LOSS flows back through the attention bias
    bias_i = log(alpha_i * r_j + 1e-6)
into the barcode maker (hash head, via alpha_i = f(h~_q, h_i) and the STE) and the keep layer (via r_j).
The LLM weights stay frozen. Loss = CE(answer) + lambda * sum(r) + beta * (auxiliary retrieval loss).

Start from the checkpoint made by train_hotpot.py, then:
    python train_e2e.py --episodes 1200            (T4 GPU: about 10-20 min)
    python llm_modulated.py --n 100 --ckpt compmem_e2e.pt --out llm_results_e2e.json
Offline test: python train_e2e.py --model <tiny dir> --data mock --n_train 800 --n_test 200 --episodes 60
"""
import argparse, json, random
import numpy as np
import torch
import torch.nn.functional as F

from train_hotpot import (load_hotpot, HashHead, talk_pool, para_text, infonce, quick_recall, gold_tensor,
                          N_PARA, N_TALK, K_RET, N_BITS)
from llm_modulated import prompt_spans, build_mask4d
from llm_link import retrieve, KeepScorer
from mih import MemoryStore
from encoder import to_int


def episode(head, keep, model, tok, e, E, texts, dev, use_bias=True, train=True):
    """One question. E: (17,d) embeddings = 16 memory chunks + the question. Returns dict of tensors."""
    head.eval()
    with torch.no_grad():                                     # retrieval uses the stored (noise-free) binary codes
        h_ev, _ = head(E)
    store = MemoryStore()
    for t, h in zip(texts, h_ev[:-1]):
        store.save(t, to_int(h))
    ids = retrieve(store, to_int(h_ev[-1]), k=K_RET)
    mu = torch.tensor([store.clusters[store.chunk_cluster[j]].mu for j in ids], device=dev)
    head.train(train)
    h, ht = head(E)                                           # differentiable: STE + Binary Concrete noise
    alpha = (1 + (h[ids] @ ht[-1]) / N_BITS) / 2              # match score  alpha_i
    r = keep(mu)                                              # keep score    r_j
    bias = torch.log(alpha * r + 1e-6) if use_bias else torch.zeros_like(r)

    p_ids, pos = prompt_spans(tok, [texts[j] for j in ids], e["question"])
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    ans = tok(e["answer"], add_special_tokens=False)["input_ids"] + [im_end]
    L, A = p_ids.size(1), len(ans)
    full = torch.cat([p_ids[0], torch.tensor(ans)])[None].to(dev)
    M = torch.zeros(len(ids), L + A, device=dev)
    for n, p in enumerate(pos):
        M[n, p.to(dev)] = 1.0
    kb = (bias[:, None] * M).sum(0)                           # per-token bias, differentiable w.r.t. alpha and r
    mask = build_mask4d(L + A, L + A, kb, model.dtype, dev)
    logits = model(input_ids=full, attention_mask=mask, use_cache=False, logits_to_keep=A + 1).logits[0, :-1]
    ce = F.cross_entropy(logits.float(), torch.tensor(ans, device=dev))

    gold = torch.zeros(N_PARA + N_TALK, dtype=torch.bool, device=dev)
    gold[list(e["gold"])] = True
    s_all = (h[:-1] * ht[-1]).sum(-1) / N_BITS
    aux = infonce(s_all[None], gold[None]) + 0.1 * (h[:-1].mean(0) ** 2).mean()
    return {"ce": ce, "r": r, "aux": aux, "recall": len(set(ids) & e["gold"]) == 2}


def val_ce(head, keep, model, tok, eps, Q, P, T, talk, dev, use_bias, n=100):
    rg, tot, rec = random.Random(77), 0.0, 0
    with torch.no_grad():
        for i in range(min(n, len(eps))):
            tk = rg.sample(range(len(talk)), N_TALK)
            E = torch.cat([P[i], T[tk], Q[i:i + 1]]).to(dev)
            texts = [para_text(t, s) for t, s in eps[i]["paras"]] + [talk[j] for j in tk]
            o = episode(head, keep, model, tok, eps[i], E, texts, dev, use_bias=use_bias, train=False)
            tot += o["ce"].item(); rec += o["recall"]
    return tot / min(n, len(eps)), rec / min(n, len(eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="compmem_hotpot.pt")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="auto"); ap.add_argument("--json", default=None)
    ap.add_argument("--n_train", type=int, default=3000); ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lr_keep", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=0.01, help="penalty lambda * sum(r)")
    ap.add_argument("--beta", type=float, default=1.0, help="weight of auxiliary retrieval loss")
    ap.add_argument("--val_n", type=int, default=100)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); random.seed(0)

    ck = torch.load(a.ckpt)
    head, keep = HashHead(ck["d_in"]), KeepScorer()
    head.load_state_dict(ck["head"]); keep.load_state_dict(ck["keep"]); head.to(dev); keep.to(dev)
    eps = load_hotpot(a.n_train + a.n_test, a.data, a.json)
    train, test = eps[:a.n_train], eps[a.n_train:a.n_train + a.n_test]
    Qtr, Ptr, Qte, Pte, T = torch.load(f"emb_{ck['embedder']}_{a.data}_{a.n_train}_{a.n_test}.pt")
    talk = talk_pool()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)                               # LLM frozen; gradients only flow THROUGH it

    # ---- baselines on held-out questions
    ce_none, _ = val_ce(head, keep, model, tok, test, Qte, Pte, T, talk, dev, use_bias=False, n=a.val_n)
    ce_before, rec_before = val_ce(head, keep, model, tok, test, Qte, Pte, T, talk, dev, use_bias=True, n=a.val_n)
    print(f"VAL answer-CE  no modulation {ce_none:.3f} | modulation, before e2e {ce_before:.3f} "
          f"(retrieval both-gold {rec_before:.2f})")

    opt = torch.optim.Adam([{"params": head.parameters(), "lr": a.lr}, {"params": keep.parameters(), "lr": a.lr_keep}])
    rg, order = random.Random(0), list(range(len(train)))
    random.Random(1).shuffle(order)
    log = {"ce": [], "qa_grad": [], "r": []}
    run_ce, run_r = [], []
    for n in range(a.episodes):
        i = order[n % len(order)]
        tk = rg.sample(range(len(talk)), N_TALK)
        E = torch.cat([Ptr[i], T[tk], Qtr[i:i + 1]]).to(dev)
        texts = [para_text(t, s) for t, s in train[i]["paras"]] + [talk[j] for j in tk]
        o = episode(head, keep, model, tok, train[i], E, texts, dev)
        if n % 25 == 0:                                       # proof that the ANSWER loss reaches the barcode maker
            gs = torch.autograd.grad(o["ce"], list(head.parameters()) + list(keep.parameters()),
                                     retain_graph=True, allow_unused=True)
            gh = float(torch.sqrt(sum((g ** 2).sum() for g in gs[:len(list(head.parameters()))] if g is not None)))
            log["qa_grad"].append((n, gh))
        loss = o["ce"] + a.lam * o["r"].sum() + a.beta * o["aux"]
        (loss / a.accum).backward()
        if (n + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(list(head.parameters()) + list(keep.parameters()), 1.0)
            opt.step(); opt.zero_grad()
        run_ce.append(o["ce"].item()); run_r.append(o["r"].mean().item())
        if (n + 1) % 50 == 0:
            log["ce"].append((n + 1, float(np.mean(run_ce)))); log["r"].append((n + 1, float(np.mean(run_r))))
            qg = log["qa_grad"][-1][1] if log["qa_grad"] else 0
            print(f"episode {n+1:5d}  answer-CE {np.mean(run_ce):.3f}  mean r {np.mean(run_r):.3f}  "
                  f"grad-norm of ANSWER loss into hash head {qg:.4f}")
            run_ce, run_r = [], []

    ce_after, rec_after = val_ce(head, keep, model, tok, test, Qte, Pte, T, talk, dev, use_bias=True, n=a.val_n)
    print(f"\nVAL answer-CE  no modulation {ce_none:.3f} | before e2e {ce_before:.3f} | AFTER e2e {ce_after:.3f}")
    print(f"VAL retrieval both-gold  before {rec_before:.2f} -> after {rec_after:.2f}")
    h_cpu = head.cpu(); both, _ = quick_recall(h_cpu, Qte, Pte, T, gold_tensor(test))
    print(f"brute-force Hamming recall@{K_RET} on all {len(test)} test questions: {both:.3f}")
    torch.save({"head": head.state_dict(), "keep": keep.state_dict(), "d_in": ck["d_in"], "embedder": ck["embedder"]},
               "compmem_e2e.pt")
    json.dump({"val_ce_no_modulation": ce_none, "val_ce_before_e2e": ce_before, "val_ce_after_e2e": ce_after,
               "val_recall_before": rec_before, "val_recall_after": rec_after, "recall_bruteforce_all_test": both,
               "qa_grad_norms": log["qa_grad"]}, open("e2e_results.json", "w"), indent=2)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(*zip(*log["ce"])); ax[0].set(title="Answer loss during end-to-end training", xlabel="episode", ylabel="CE")
    ax[1].plot(*zip(*log["qa_grad"]), "o-"); ax[1].set(title="Grad norm of the ANSWER loss into the hash head (> 0 = end-to-end works)",
                                                      xlabel="episode", yscale="log")
    plt.tight_layout(); plt.savefig("e2e_curves.png", dpi=150)
    print("saved compmem_e2e.pt, e2e_results.json, e2e_curves.png")


if __name__ == "__main__":
    main()
