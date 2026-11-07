"""Phase 3, Step C: train the encoder + keep layer (+ tiny reader) on a multi-hop QA task.

NOTE: no internet / HuggingFace here, so this uses a SYNTHETIC multi-hop QA set:
    "alice works at zenith."  +  "zenith is located in paris."
    Q: "In which city is the company where alice works located?"  ->  paris
mixed with distractor chains and small talk. To use HotpotQA/2Wiki later, replace make_episode().
Loss = CrossEntropy(answer) + lambda * sum(r_j)   (penalty pushes useless chunks to r_j -> 0)
"""
import random
import torch
import torch.nn.functional as F

from encoder import HashEncoder, to_int
from mih import MemoryStore
from llm_link import KeepScorer, ModulatedReader, retrieve, build_memory_inputs
from chunker import token_ids

PEOPLE = "alice bob carol dave erin frank grace heidi ivan judy kate liam mona nick olga paul quinn rita sam tina".split()
COMPS = "zenith apex nimbus orbit vertex quasar helix pylon summit tundra lumen cobalt ember falcon glacier harbor indigo jasper kestrel lotus".split()
CITIES = "paris tokyo cairo lima oslo delhi rome miami".split()
TALK = ["Hello, how are you?", "Good morning!", "Thanks, see you later.", "Nice weather today.",
        "I am fine, thanks.", "Please remind me about the exam."]
LAMBDA, K = 0.002, 4


def make_episode(rng):
    ppl, cmp_, cty = rng.sample(PEOPLE, 3), rng.sample(COMPS, 3), rng.sample(CITIES, 3)
    facts = []
    for p, c, y in zip(ppl, cmp_, cty):
        facts += [f"{p} works at {c}.", f"{c} is located in {y}."]
    chunks = facts + rng.sample(TALK, 2)
    rng.shuffle(chunks)
    q = f"In which city is the company where {ppl[0]} works located?"
    relevant = {chunks.index(facts[0]), chunks.index(facts[1])}
    return chunks, q, CITIES.index(cty[0]), relevant


def run_episode(enc, keep, reader, ep, train=True):
    chunks, q, ans, relevant = ep
    # memory codes: binary, no grad (stored barcodes)
    h_mem = enc.encode_texts(chunks)                                  # (n,64) +-1
    store, h_of = MemoryStore(), {}
    for t, h in zip(chunks, h_mem):
        cid = store.save(t, to_int(h))
        h_of[cid] = h
    # query: smooth barcode h~_q (grad flows to encoder)
    q_code = to_int(enc.encode_texts([q])[0])                          # noise-free query code for lookup
    enc.train(train)
    _, ht_all = enc(enc.batch_ids(chunks + [q]))                       # smooth codes, with grad
    ht_mem, h_tilde_q = ht_all[:-1], ht_all[-1]
    ids = retrieve(store, q_code, k=K)
    mu_of = lambda st, i: torch.tensor(st.clusters[st.chunk_cluster[i]].mu)
    mem_ids, bias, r = build_memory_inputs(store, ids, h_tilde_q, h_of, mu_of, keep)
    logits = reader(torch.tensor(token_ids(q)), mem_ids, bias)
    loss = F.cross_entropy(logits[None], torch.tensor([ans])) + LAMBDA * r.sum()
    if train:   # AUX retrieval loss (deviation from SOW): supporting-fact labels teach the encoder
        labels = torch.tensor([1.0 if i in relevant else 0.0 for i in range(len(chunks))])
        a_all = ((1 + ht_mem @ h_tilde_q / h_tilde_q.numel()) / 2).clamp(1e-4, 1 - 1e-4)
        loss = loss + F.binary_cross_entropy(a_all, labels)
    recall = relevant <= set(ids)                                     # both needed facts retrieved?
    return loss, int(logits.argmax()) == ans, recall, r.detach(), ids, relevant


def main(steps=2000, seed=0):
    random.seed(seed); torch.manual_seed(seed)
    rng = random.Random(seed)
    enc, keep, reader = HashEncoder(), KeepScorer(), ModulatedReader(len(CITIES))
    params = list(enc.parameters()) + list(keep.parameters()) + list(reader.parameters())
    opt = torch.optim.Adam(params, lr=2e-3)
    run_loss = run_acc = run_rec = 0.0
    for s in range(1, steps + 1):
        loss, ok, rec, *_ = run_episode(enc, keep, reader, make_episode(rng))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        run_loss += loss.item(); run_acc += ok; run_rec += rec
        if s % 250 == 0:
            print(f"step {s:5d}  loss {run_loss/250:.3f}  answer-acc {run_acc/250:.2f}  "
                  f"retrieval-recall {run_rec/250:.2f}")
            run_loss = run_acc = run_rec = 0.0
    # held-out evaluation
    enc.eval(); test_rng = random.Random(999)
    acc = rec = 0; r_rel, r_irr = [], []
    with torch.no_grad():
        for _ in range(300):
            ep = make_episode(test_rng)
            _, ok, rc, r, ids, relevant = run_episode(enc, keep, reader, ep, train=False)
            acc += ok; rec += rc
            for n, i in enumerate(ids):
                (r_rel if i in relevant else r_irr).append(r[n].item())
    print(f"\nTEST (300 episodes): answer-acc {acc/300:.2f} | retrieval-recall {rec/300:.2f} "
          f"(random baseline recall = 0.21, answer chance = 0.125)")
    print(f"mean keep-score r: relevant chunks {sum(r_rel)/max(len(r_rel),1):.3f} | "
          f"irrelevant chunks {sum(r_irr)/max(len(r_irr),1):.3f}")
    torch.save({"enc": enc.state_dict(), "keep": keep.state_dict(), "reader": reader.state_dict()},
               "compmem_ckpt.pt")


if __name__ == "__main__":
    main()
