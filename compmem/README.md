# CompMem – Compressive Memory for Long-Term AI Chatbots

IDEAS-TIH Autumn Internship 2026 · ISI Kolkata · Venushree Gayathri C U

Standard LLMs slow down quadratically and need gigabytes of KV-cache on long conversations. CompMem cuts the
conversation into ≤256-token chunks, turns each chunk into a **64-bit ±1 barcode**, finds relevant memories with
**Multi-Index Hashing**, feeds only those to the LLM with **soft attention modulation**, and a background
**cleaner** keeps memory bounded.

## Pipeline and files

| Phase | File | What it does |
|---|---|---|
| 1A | `chunker.py` | Splits text into chunks of ≤256 tokens |
| 1B/1C | `encoder.py`, `train_hotpot.py` | Transformer embedding → hash head → Binary Concrete relaxation + straight-through estimator → ±1 code |
| 2 | `mih.py` | 4 hash tables of 16-bit pieces, multi-probe search (Hamming ≤ 7), medoid clustering (≤ 4 bits) |
| 3A | `llm_link.py` | Match score α, keep score r |
| 3B | `llm_modulated.py` | Adds `log(α·r + 1e-6)` to a real LLM's attention scores through a custom 4-D attention mask |
| 3C | `train_hotpot.py` | Trains on HotpotQA (distractor) with supporting-fact labels; logs gradient norms |
| 4A | `cleaner.py` | Async cleaner: delete clusters with r < 0.25; LLM-summarize clusters with > 5 chunks when memory > 512 |
| 4B | `benchmark.py` | RAM and time vs a standard full-attention LLM up to 100k tokens |
| tests | `demo.py`, `test_phase4.py`, `llm_modulated.py --selftest` | Small sanity tests (20 sentences, 620-chunk cleaner test, bias mechanism) |
| old | `train.py`, `benchmark` toy files | Early synthetic-data version, kept for reference |

## How to run

```bash
pip install torch transformers sentence-transformers datasets scikit-learn matplotlib numpy
python demo.py && python test_phase4.py && python llm_modulated.py --selftest   # sanity tests
python train_hotpot.py --n_train 3000 --n_test 500                               # train on HotpotQA
python llm_modulated.py --n 100                                                  # real LLM evaluation (GPU recommended)
python benchmark.py                                                              # RAM/speed plot
```
Step-by-step with timings: `RUN_ME.md`.

## Results

*(Fill these in from `results.json`, `llm_results.json`, `benchmark.png`.)*

| Metric | Value |
|---|---|
| Retrieval: both gold paragraphs in top-4 (CompMem codes) | _from results.json_ |
| Random baseline | 0.05 |
| Keep score r: gold / distractor / small-talk | _from results.json_ |
| Answer F1: full context / CompMem modulated / CompMem no modulation / closed book | _from llm_results.json_ |
| Prompt tokens: full context vs CompMem | _from llm_results.json_ |
| RAM at 100k tokens: standard KV-cache vs CompMem | see `benchmark.png` |

`training_curves.png` shows the loss, the gradient norm through the STE (stays > 0), and test recall per epoch.

## Design notes and deviations from the SOW

* **STE:** `h = sign(h~) + (h~ - h~.detach())`, bracketed so the forward value is exactly ±1.
* **Pigeonhole search:** Hamming ≤ 7 over 4 pieces ⇒ some piece differs by ≤ 1 bit; probing the exact piece + 16 one-bit flips per table is exact (verified against brute force).
* **Encoder:** a frozen pretrained MiniLM Transformer feeds a trainable hash head (the Binary Concrete + STE part); training a Transformer from scratch needs far more data than the 2-hour scope allows.
* **Auxiliary loss:** the SOW's loss gives the encoder no direct retrieval signal (the lookup is non-differentiable), so an InfoNCE loss on the barcode match using HotpotQA supporting facts and a bit-balance penalty are added.
* **Keep layer:** trained with supporting-fact labels plus λ·Σr with the LLM frozen.
* **Modulation:** the bias is added to every layer and head for the tokens of retrieved chunks. Since α·r ≤ 1, the bias is ≤ 0: weak matches are suppressed, perfect matches are unchanged.
* **Benchmark:** CompMem is measured; standard attention time is measured up to the stated length and extrapolated with a quadratic fit beyond; KV-cache size is computed from the model config.

## Limitations

* Evaluated on HotpotQA dev with a 0.5B model and a small number of questions; results are indicative, not a full benchmark.
* HotpotQA episodes are short (10 paragraphs + small talk), so the 100k-token numbers come from the benchmark script, not from a real 100k-token conversation.
