# FINISH CompMem IN 2 HOURS – do these steps in order

Use **Google Colab** (free GPU): colab.research.google.com → New notebook → Runtime → Change runtime type → **T4 GPU**.
(Your laptop works for steps 1-3 and 5, but step 4 (the real LLM) needs a GPU or will be slow.)

## Timeline

| Time | Step |
|---|---|
| 0:00-0:10 | Step 0: upload + install |
| 0:10-0:20 | Step 1: sanity tests |
| 0:20-0:50 | Step 2: train on real HotpotQA |
| 0:50-1:15 | Step 3: real LLM evaluation |
| 1:15-1:30 | Step 4: benchmark plot |
| 1:30-1:50 | Step 5: GitHub + README numbers |
| 1:50-2:00 | Step 6: email Sir |

## Step 0 – Upload and install (Colab cells; put `!` in front of each line)

1. Click the folder icon on the left → upload `compmem.zip` (the zip I sent).
2. Run:
```
!unzip -q compmem.zip -d . && cd compmem && pip install -q sentence-transformers datasets scikit-learn matplotlib
```
Every command after this must be run **inside the folder**, so start each cell with `%cd /content/compmem`.

## Step 1 – Sanity tests (2 min)
```
%cd /content/compmem
!python demo.py
!python test_phase4.py
!python llm_modulated.py --selftest
```
All three must end with "ALL CHECKS PASSED" / "PHASE 4 CLEANER OK" / "SELFTEST OK".

## Step 2 – Train on real HotpotQA (Phase 3C)
```
%cd /content/compmem
!python train_hotpot.py --n_train 3000 --n_test 500 --epochs 30
```
It downloads HotpotQA itself. If it fails with a download error, download
`hotpot_dev_distractor_v1.json` from hotpotqa.github.io, upload it, and add `--json hotpot_dev_distractor_v1.json`.

Takes about 5-15 minutes. **Success** = the last lines show *"both-gold recall@4"* clearly **above the random baseline**
(random ≈ 0.05). Outputs: `results.json`, `training_curves.png` (loss + gradient norms: show this to Sir), `compmem_hotpot.pt`.

If recall is not clearly above random: try `--lr 1e-3 --epochs 50`, or `--balance 3`.
Do not worry about the exact value, but write down whatever number you get.

## Step 3 – Real LLM with attention modulation (Phase 3B + 4A summarizer)
```
%cd /content/compmem
!python llm_modulated.py --n 100
```
Uses Qwen2.5-0.5B-Instruct (downloads automatically). Prints a table comparing:
closed book / full context (standard LLM) / random 4 chunks / CompMem without modulation / **CompMem with modulation** / oracle gold.
It also prints a real LLM summary used by the cleaner. Output: `llm_results.json`.

Write down honestly whatever comes out. A good result is: CompMem close to full context while using far fewer prompt tokens.
Modulation helping a little, or not at all, is still a valid finding for a 0.5B model; report it as measured.

## Step 4 – RAM and speed plot (Phase 4B)
```
%cd /content/compmem
!python benchmark.py
```
Output: `benchmark.png` (standard model measured up to 32k tokens on GPU, then extrapolated; CompMem measured up to 100k).

## Step 5 – GitHub (the SOW asks for it)

1. Download from Colab's file panel: all `.py` files, `README.md`, `RUN_ME.md`, `results.json`, `llm_results.json`, `training_curves.png`, `benchmark.png`.
   Do not upload `.pt` files or `emb_*.pt` (large).
2. On github.com click New repository → name it `CompMem` → Public → Create.
3. Click "uploading an existing file", drag in the files, and click Commit.
4. Open `README.md` and fill in the **Results** table with your real numbers from `results.json` and `llm_results.json`.

## Step 6 – Email Sir

Send the GitHub link plus the results table and `training_curves.png`, and `benchmark.png`.
Ask me for the email draft once you have your numbers and I'll write it with them.

## Things to tell Sir (deviations from the SOW)

1. The encoder uses a **frozen pretrained MiniLM Transformer** and trains the hash head (MLP + Binary Concrete + STE) on top. Training the whole Transformer from scratch needs far more data/time.
2. The loss uses an **auxiliary retrieval loss** (InfoNCE with HotpotQA supporting facts) plus a bit-balance penalty, because the SOW's loss gives the encoder no direct retrieval signal.
3. The keep layer is trained with supporting-fact labels plus the λ·Σr penalty, with the LLM frozen.
4. The attention bias is added to all layers and heads via a custom attention mask. Because α·r ≤ 1, the log is ≤ 0, so it down-weights weak matches and leaves perfect matches unchanged.
5. Standard-LLM time beyond the measured range is extrapolated; KV-cache size is computed from the model config.
