"""Phase 3B on a REAL LLM + Phase 4A summarizer.

Soft KV-Modulation: for every token that belongs to a retrieved chunk i we ADD  log(alpha_i * r_j + 1e-6)
to the attention score (pre-softmax) of every layer/head. This is done through a custom 4-D additive
attention mask, so no change to the model weights/code is needed.

Run on Colab GPU:   python llm_modulated.py --n 100
Offline mechanism test (tiny random model, no downloads):   python llm_modulated.py --selftest
"""
import argparse, json, re, random, string, time
from collections import Counter
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ core: forward / greedy decode with per-key bias
def build_mask4d(L_q, L_k, key_bias, dtype, device):
    """Additive 4-D mask (1,1,L_q,L_k): causal, plus key_bias[k] on every allowed (q,k) pair.
    The queries are the LAST L_q positions of the L_k keys (works for prefill and cached decoding)."""
    neg = torch.finfo(dtype).min
    qpos = torch.arange(L_k - L_q, L_k, device=device).unsqueeze(1)
    kpos = torch.arange(L_k, device=device).unsqueeze(0)
    allowed = kpos <= qpos
    bias = key_bias.to(device=device, dtype=dtype).unsqueeze(0).expand(L_q, L_k)
    return torch.where(allowed, bias, torch.full_like(bias, neg))[None, None]


@torch.no_grad()
def forward_with_bias(model, input_ids, key_bias):
    L = input_ids.size(1)
    mask = build_mask4d(L, L, key_bias, model.dtype, input_ids.device)
    return model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits


@torch.no_grad()
def generate_with_bias(model, input_ids, key_bias, max_new=24, eos_ids=()):
    """Greedy decoding with KV-cache; the bias stays attached to the original prompt keys."""
    dev, dt = input_ids.device, model.dtype
    L = input_ids.size(1)
    kb = key_bias.clone()
    out = model(input_ids=input_ids, attention_mask=build_mask4d(L, L, kb, dt, dev), use_cache=True)
    past, nxt, gen = out.past_key_values, out.logits[0, -1].argmax().item(), []
    for _ in range(max_new):
        if nxt in eos_ids:
            break
        gen.append(nxt)
        kb = torch.cat([kb, torch.zeros(1, device=kb.device, dtype=kb.dtype)])   # new tokens: no bias
        step = model(input_ids=torch.tensor([[nxt]], device=dev),
                     attention_mask=build_mask4d(1, kb.numel(), kb, dt, dev),
                     past_key_values=past, use_cache=True)
        past, nxt = step.past_key_values, step.logits[0, -1].argmax().item()
    return gen


# ------------------------------------------------------------------ prompt building with exact chunk token spans
SYS = "Answer the question using the context. Reply with only a short answer (a few words), no explanation."


def prompt_spans(tok, chunks, question):
    """Returns input_ids (1,L) and, for every chunk, the LongTensor of its token positions."""
    user, spans = "Context:\n", []
    for c in chunks:
        s = len(user); user += c; spans.append((s, len(user))); user += "\n\n"
    user += f"Question: {question}\nAnswer:"
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    shift = text.index(user)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
    offs = enc["offset_mapping"][0].tolist()
    pos = [torch.tensor([t for t, (a, z) in enumerate(offs) if a >= s + shift and z <= e + shift and z > a],
                        dtype=torch.long) for (s, e) in spans]
    return enc["input_ids"], pos


def build_prompt(tok, chunks, question, biases):
    """chunks: list[str]; biases: list[float] (log(alpha*r)). Returns input_ids (1,L), key_bias (L,)."""
    ids, pos = prompt_spans(tok, chunks, question)
    kb = torch.zeros(ids.size(1))
    for p, b in zip(pos, biases):
        kb[p] = float(b)
    return ids, kb


def answer(model, tok, chunks, question, biases):
    ids, kb = build_prompt(tok, chunks, question, biases)
    dev = next(model.parameters()).device
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None}
    gen = generate_with_bias(model, ids.to(dev), kb.to(dev), eos_ids=eos)
    return tok.decode(gen, skip_special_tokens=True).strip(), ids.size(1)


# ------------------------------------------------------------------ HotpotQA metrics
def norm_ans(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())


def f1_em(pred, gold):
    p, g = norm_ans(pred).split(), norm_ans(gold).split()
    em = float(p == g)
    common = Counter(p) & Counter(g); n = sum(common.values())
    if n == 0:
        return 0.0, em
    pr, rc = n / len(p), n / len(g)
    return 2 * pr * rc / (pr + rc), em


# ------------------------------------------------------------------ LLM summarizer for the cleaner (Phase 4A, Rule 2)
def make_summarizer(model, tok):
    def summarize(texts):
        msgs = [{"role": "user", "content": "Summarize these notes into ONE short paragraph, keeping every "
                 "distinct fact:\n\n" + "\n".join(f"- {t}" for t in texts)}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        dev = next(model.parameters()).device
        out = model.generate(**{k: v.to(dev) for k, v in ids.items()}, max_new_tokens=80, do_sample=False)
        return tok.decode(out[0, ids["input_ids"].size(1):], skip_special_tokens=True).strip()
    return summarize


# ------------------------------------------------------------------ self test (no downloads)
def selftest():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    model = Qwen2ForCausalLM(cfg).eval()
    ids = torch.randint(5, 200, (1, 40))
    zero = torch.zeros(40)
    ref = model(input_ids=ids).logits                                           # standard causal forward
    got = forward_with_bias(model, ids, zero)
    print("1) bias=0 equals standard forward:        max diff", (ref - got).abs().max().item())
    assert (ref - got).abs().max() < 1e-4
    kb = zero.clone(); kb[10:20] = torch.log(torch.tensor(0.3))
    diff = (forward_with_bias(model, ids, kb)[0, -1] - ref[0, -1]).abs().max().item()
    print("2) bias on tokens 10-19 changes output:   max diff", diff)
    assert diff > 1e-4
    kb2 = zero.clone(); kb2[10:20] = -1e4                                         # huge negative = tokens invisible
    dropped = forward_with_bias(model, ids, kb2)[0, -1]
    ids_removed = torch.cat([ids[:, :10], ids[:, 20:]], 1)
    # removing tokens also shifts positions, so compare with masking only (sanity: differs from reference)
    print("3) strong negative bias hides tokens:     differs from ref by", (dropped - ref[0, -1]).abs().max().item())
    g_cache = generate_with_bias(model, ids, kb, max_new=8)
    seq, g_ref = ids.clone(), []
    for _ in range(8):                                                            # no-cache recompute
        full = torch.cat([kb, torch.zeros(seq.size(1) - kb.numel())]) if seq.size(1) > kb.numel() else kb
        lg = forward_with_bias(model, seq, full)[0, -1].argmax().item()
        g_ref.append(lg); seq = torch.cat([seq, torch.tensor([[lg]])], 1)
    print("4) cached decoding == recompute decoding:", g_cache == g_ref, g_cache)
    assert g_cache == g_ref
    print("SELFTEST OK")


# ------------------------------------------------------------------ evaluation on HotpotQA
def evaluate(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train_hotpot import (load_hotpot, Embedder, HashHead, talk_pool, para_text, N_PARA, N_TALK, K_RET)
    from mih import MemoryStore
    from encoder import to_int
    from llm_link import retrieve, match_score, KeepScorer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt)
    head, keep = HashHead(ck["d_in"]), KeepScorer()
    head.load_state_dict(ck["head"]); keep.load_state_dict(ck["keep"]); head.eval()
    emb = Embedder(ck["embedder"], device=dev)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16 if dev == "cuda" else torch.float32,
                                                 attn_implementation="sdpa").to(dev).eval()
    eps = load_hotpot(a.n_train + a.n_test_pool, a.data, a.json)[a.n_train:a.n_train + a.n]   # held-out questions
    talk, rg = talk_pool(), random.Random(3)
    conds = ["closed_book", "full_context(10 paras)", "random_4", "compmem_no_modulation", "compmem_modulated", "oracle_gold"]
    res = {c: {"f1": [], "em": [], "tokens": [], "sec": []} for c in conds}
    rec = []
    for n, e in enumerate(eps):
        tk = rg.sample(talk, N_TALK)
        texts = [para_text(t, s) for t, s in e["paras"]] + tk
        with torch.no_grad():
            E = emb(texts + [e["question"]]); h, ht = head(E)
        codes = [to_int(x) for x in h[:-1]]
        store = MemoryStore()
        for t, c in zip(texts, codes):
            store.save(t, c)
        ids = retrieve(store, to_int(h[-1]), k=K_RET)
        rec.append(len(set(ids) & e["gold"]) == 2)
        with torch.no_grad():
            alpha = match_score(ht[-1], h[ids])
            mu = torch.tensor([store.clusters[store.chunk_cluster[i]].mu for i in ids])
            r = keep(mu)
            bias = torch.log(alpha * r + 1e-6).tolist()
        sel = [texts[i] for i in ids]
        runs = {"closed_book": ([], []), "full_context(10 paras)": (texts[:N_PARA], [0.0] * N_PARA),
                "random_4": ([texts[i] for i in rg.sample(range(len(texts)), K_RET)], [0.0] * K_RET),
                "compmem_no_modulation": (sel, [0.0] * K_RET), "compmem_modulated": (sel, bias),
                "oracle_gold": ([texts[i] for i in sorted(e["gold"])], [0.0, 0.0])}
        for c, (ch, bs) in runs.items():
            t0 = time.time(); pred, ntok = answer(model, tok, ch, e["question"], bs)
            f1, em = f1_em(pred, e["answer"])
            for k_, v in (("f1", f1), ("em", em), ("tokens", ntok), ("sec", time.time() - t0)):
                res[c][k_].append(v)
        if (n + 1) % 20 == 0:
            print(f"{n+1}/{len(eps)} done")
    mean = lambda x: sum(x) / len(x)
    table = {c: {k: mean(v) for k, v in d.items()} for c, d in res.items()}
    table["retrieval_both_gold_recall"] = mean(rec)
    print(f"\n{'condition':28s} {'F1':>6s} {'EM':>6s} {'prompt tok':>11s} {'sec/q':>7s}")
    for c in conds:
        t = table[c]; print(f"{c:28s} {t['f1']:6.3f} {t['em']:6.3f} {t['tokens']:11.0f} {t['sec']:7.2f}")
    print("retrieval both-gold recall:", round(table["retrieval_both_gold_recall"], 3))
    json.dump(table, open(a.out, "w"), indent=2)

    # Phase 4A with a real LLM summarizer
    summ = make_summarizer(model, tok)
    notes = ["The exam is next Friday.", "Exam date: Friday next week.", "Reminder: the exam is on Friday.",
             "Please remind me, exam on Friday.", "Exam Friday, revise graphs.", "Don't forget the Friday exam."]
    print("\nSummarizer demo:", summ(notes))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="auto")
    ap.add_argument("--json", default=None)
    ap.add_argument("--n", type=int, default=100, help="number of held-out test questions")
    ap.add_argument("--n_train", type=int, default=3000, help="must match train_hotpot.py")
    ap.add_argument("--ckpt", default="compmem_hotpot.pt")
    ap.add_argument("--out", default="llm_results.json")
    ap.add_argument("--n_test_pool", type=int, default=500, help="must match train_hotpot.py --n_test")
    a = ap.parse_args()
    selftest() if a.selftest else evaluate(a)
