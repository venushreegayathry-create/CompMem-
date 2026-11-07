# CompMem --- Setup, Execution, and Submission Guide

This guide explains how to set up the CompMem project, run its tests and
experiments, generate the evaluation outputs, and prepare the repository
for submission.

## 1. Environment Setup

The recommended environment is **Google Colab with a T4 GPU**. Steps
1--3 and 5 can also run on a local machine, but running the
language-model evaluation (Step 3) is likely to be slow without a GPU.

1.  Open [Google Colab](https://colab.research.google.com/) and create a
    new notebook.
2.  Select **Runtime → Change runtime type → T4 GPU**.
3.  Upload `compmem.zip` using the Files panel on the left.
4.  Extract the project and install the required dependencies:

``` bash
!unzip -q compmem.zip -d .
%cd /content/compmem
!pip install -q sentence-transformers datasets scikit-learn matplotlib
```

Run the commands in the remaining sections from the project directory.
In Colab, begin each new cell with:

``` python
%cd /content/compmem
```

## 2. Run the Sanity Tests

Run the following commands:

``` bash
!python demo.py
!python test_phase4.py
!python llm_modulated.py --selftest
```

Check that the output indicates successful completion:

-   `demo.py`: `ALL CHECKS PASSED`
-   `test_phase4.py`: `PHASE 4 CLEANER OK`
-   `llm_modulated.py --selftest`: `SELFTEST OK`

Resolve any errors before proceeding to the experiments.

## 3. Train and Evaluate on HotpotQA

This step runs Phase 3C training using the HotpotQA dataset.

``` bash
!python train_hotpot.py --n_train 3000 --n_test 500 --epochs 30
```

The script attempts to download HotpotQA automatically. If the download
fails, download `hotpot_dev_distractor_v1.json` from the [HotpotQA
website](https://hotpotqa.github.io/), upload the file to the Colab
session, and run:

``` bash
!python train_hotpot.py --n_train 3000 --n_test 500 --epochs 30 --json hotpot_dev_distractor_v1.json
```

The run may take approximately 5--15 minutes, depending on the runtime
environment.

**Evaluation check:** Inspect the reported `both-gold recall@4` value
and compare it with the random baseline (approximately 0.05). Record the
actual value produced by the run; do not substitute an expected or
target value.

If the result is not clearly above the random baseline, the following
are possible settings to test:

``` bash
!python train_hotpot.py --n_train 3000 --n_test 500 --epochs 50 --lr 1e-3
```

Alternatively, try adding `--balance 3` to the original command. Record
which settings were used and report the resulting measurements
accurately.

Expected output files:

-   `results.json`
-   `training_curves.png`
-   `compmem_hotpot.pt`

## 4. Evaluate with a Language Model

This step runs the Phase 3B evaluation and Phase 4A summarizer using
Qwen2.5-0.5B-Instruct.

``` bash
!python llm_modulated.py --n 100
```

The model is downloaded automatically when required. The evaluation
compares:

1.  Closed-book answering
2.  Full-context answering (standard LLM)
3.  Random selection of four chunks
4.  CompMem without attention modulation
5.  CompMem with attention modulation
6.  Oracle gold context

The script also generates an LLM summary used by the cleaner.

Expected output file:

-   `llm_results.json`

Record the measured results as they appear. In particular, compare
CompMem's performance with the full-context baseline and consider the
number of prompt tokens used. Attention modulation may provide only a
small improvement, or no improvement, with a 0.5B-parameter model;
report the observed result rather than assuming an improvement.

## 5. Generate the RAM and Speed Benchmark

Run:

``` bash
!python benchmark.py
```

Expected output file:

-   `benchmark.png`

**Benchmark note:** The standard-model measurements are taken up to 32k
tokens on GPU and then extrapolated. CompMem is measured up to 100k
tokens. Clearly distinguish measured values from extrapolated values
when discussing the plot.

## 6. Prepare the GitHub Repository

Create a repository for the project on GitHub.

1.  Open [GitHub](https://github.com/) and create a new repository named
    `CompMem`.
2.  Set the repository visibility according to the submission
    requirements (for example, Public if a public repository is
    required).
3.  Upload the project source files and documentation.
4.  Include the generated evaluation outputs listed below.
5.  Commit the files and verify that they appear correctly in the
    repository.

### Recommended files to include

-   All project `.py` source files
-   `README.md`
-   `RUN_ME.md`
-   `results.json`
-   `llm_results.json`
-   `training_curves.png`
-   `benchmark.png`

### Files not to upload

Do not upload model-weight or embedding files such as:

-   `*.pt`
-   `emb_*.pt`

These files can be large. The scripts should generate the required
artifacts when the project is run.

### Update the results in README.md

Before submission, fill in the Results section of `README.md` using the
actual values from `results.json` and `llm_results.json`.

Also ensure that:

-   The reported dataset sizes and training settings match the run you
    performed.
-   The plots correspond to the reported experiment.
-   Any failed, incomplete, or repeated runs are not presented as
    successful results.
-   Measured and extrapolated benchmark values are clearly
    distinguished.

## 7. Submission to the Faculty

After the repository is complete, share the GitHub repository link with
the faculty member along with:

-   A concise summary of the project and its results
-   The completed results table from `README.md`
-   `training_curves.png`
-   `benchmark.png`

Check that the repository is accessible to the intended reviewer before
sharing the link.

## 8. Implementation Notes and Deviations from the Statement of Work

The following implementation details should be disclosed when presenting
the project:

1.  **Encoder:** The encoder uses a frozen, pretrained MiniLM
    Transformer. The hash head (MLP + Binary Concrete + Straight-Through
    Estimator) is trained on top of it. Training the entire Transformer
    from scratch would require substantially more data and compute time.
2.  **Training loss:** The loss includes an auxiliary retrieval loss
    (InfoNCE using HotpotQA supporting facts) and a bit-balance penalty.
    This provides the encoder with a direct retrieval signal.
3.  **Keep layer:** The keep layer is trained using supporting-fact
    labels and the `λ·Σr` penalty, while the LLM remains frozen.
4.  **Attention bias:** The attention bias is applied across all layers
    and heads using a custom attention mask. Since `α·r ≤ 1`, its
    logarithm is non-positive; therefore, it down-weights weaker matches
    while leaving perfect matches unchanged.
5.  **Benchmarking:** Standard-LLM time beyond the measured range is
    extrapolated. KV-cache size is calculated from the model
    configuration.

These details should be described transparently in the project
documentation and final submission.
