# Source

This repository is a small research codebase for three connected tasks:

1. Run code-generation and self-debug evaluation on LiveCodeBench.
2. Cluster generated solutions and compute entropy-style uncertainty metrics.
3. Compare simple baselines and a cascade-style test-time scaling pipeline.

The examples below assume you run commands from the parent directory that contains the `source/` package directory.

## Modules

### `source/live_code_bench`

This package handles dataset loading, prompting, code generation, debugging, and test execution.

- `LCB_bench.py`: loads the LiveCodeBench dataset from Hugging Face, decodes private tests, and converts records into DSPy examples.
- `DSPY_PROMPT.py`: defines DSPy signatures for code generation, self-debugging, and test generation.
- `LCB_run.py`: runs generation and self-debug loops, extracts token/logprob metadata, and writes per-problem results.
- `Test_Utils.py`: executes generated Python against public and private tests with subprocess/time-limit protection.
- `utils.py`: post-processes model code and formats test feedback.
- `interface.py`: command-line entry point for single-round or multi-round evaluation.

### `source/ESE`

This package computes clustering-based uncertainty signals and evaluates them.

- `calc_entropy.py`: loads saved model runs, clusters solutions, and computes `PE_MC`, `PE_Rao`, `SE`, and `DSE`.
- `evaluate.py`: computes AUROC and correlation metrics and generates plots/summaries.
- `entropy/semantic_entropy.py`: core entropy formulas.
- `clustering/`: clustering backends.

Available clustering methods are:

- `embed`: embedding similarity with `sentence-transformers`
- `bleu`: CodeBLEU similarity
- `nlg_deberta`: NLI clustering with DeBERTa
- `nlg_llm`: NLI clustering with an LLM
- `symbolic`: symbolic equivalence checking
- `functional`: clustering via generated functional tests
- `functional_vanilla`: simpler random-test functional clustering

### `source/TTS`

This package consumes saved LiveCodeBench runs and studies selection strategies.

- `baselines.py`: baseline selectors such as `vanilla`, `major_voting`, `pass_at_n`, and `pass_at_n_oneshot`.
- `cascade_tts.py`: multi-layer cascade selection using entropy and public-test failure signals.

## Installation

From the parent directory of `source/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r source/requirements.txt
```

## Configuration

### 1. Model config for `live_code_bench`

Create a JSON config file, for example `source/live_code_bench/config.json`:

```json
{
  "generate_lm": {
    "model_name": "openai/qcoder",
    "api_base": "http://localhost:8001/v1",
    "api_key": "your-api-key",
    "timeout": 120,
    "temperature": 0.7,
    "max_tokens": 8192,
    "cache": true
  }
}
```

`interface.py` requires the `generate_lm` object. Extra DSPy/litellm-compatible fields can also be added.

### 2. Layer config for `TTS`

Create a JSON file, for example `source/TTS/config.json`:

```json
{
  "layers_count": 2,
  "layers": [
    {
      "layer_idx": 0,
      "samples": {
        "openai_qcoder": [0, 1, 2, 3, 4]
      },
      "alpha": 1.0,
      "beta": 1.0,
      "threshold": 0.5
    },
    {
      "layer_idx": 1,
      "samples": {
        "openai_qcoder": [5, 6, 7, 8, 9]
      },
      "alpha": 1.0,
      "beta": 1.0,
      "threshold": 0.2
    }
  ]
}
```

Notes:

- `samples` is a mapping from saved model directory name to a list of round indices.
- `cascade_tts.py` uses all configured layers.
- `baselines.py` only uses the first layer.

### 3. Question metadata for `ESE` and `TTS`

Some downstream scripts expect a question metadata file such as `lcb_release_v2_all_questions.json`.

It should be a JSON array of question objects. The code will index that array by `task_id`, and each object should include at least:

- `task_id`
- `prompt`
- `is_stdin`

If you already have the official LiveCodeBench metadata file used during your experiments, point the scripts to that file directly.

## Recommended Workflow

### Step 1. Generate LiveCodeBench runs

Single round:

```bash
python3 -m source.live_code_bench.interface \
  --model_name openai/qcoder \
  --iter 0 \
  --config_path source/live_code_bench/config.json \
  --dataset_difficulty easy \
  --num_rounds 3 \
  --num_workers 4
```

Full 20-round sweep:

```bash
python3 -m source.live_code_bench.interface \
  --model_name openai/qcoder \
  --full \
  --config_path source/live_code_bench/config.json \
  --dataset_difficulty easy \
  --num_rounds 3 \
  --num_workers 4
```

Output is written under:

```text
source/live_code_bench/result/<difficulty>/<safe_model_name>/
```

`safe_model_name` is the `--model_name` value with `/` and `\` replaced by `_`.

Important:

- `interface.py` currently writes line-delimited JSON content but names files `round_<n>.json`.
- `ESE` and `TTS` expect files named `round_<n>.jsonl`.
- Before running downstream steps, rename the files to `.jsonl`, or adjust the filename in `source/live_code_bench/interface.py`.

### Step 2. Compute entropy and clustering results

Example with embedding clustering:

```bash
python3 -m source.ESE.calc_entropy \
  --model openai_qcoder \
  --difficulty easy \
  --method embed \
  --rounds 0,5,10,15,19 \
  --questions-json lcb_release_v2_all_questions.json \
  --result-dir source/live_code_bench/result \
  --output-dir source/ESE/result
```

This writes:

```text
source/ESE/result/<difficulty>/<model>/<method>/result.jsonl
```

### Step 3. Evaluate entropy metrics

```bash
python3 -m source.ESE.evaluate \
  --result-dir source/ESE/result \
  --output-dir source/ESE/evaluation
```

This produces AUROC summaries, correlation summaries, and ROC plots.

### Step 4. Run baseline selectors

```bash
python3 -m source.TTS.baselines \
  --config source/TTS/config.json \
  --result_base source/live_code_bench/result \
  --difficulty easy \
  --method major_voting \
  --clustering_method embed \
  --questions_json lcb_release_v2_all_questions.json
```

### Step 5. Run cascade TTS

```bash
python3 -m source.TTS.cascade_tts \
  --config source/TTS/config.json \
  --result_base source/live_code_bench/result \
  --difficulty easy \
  --clustering_method embed \
  --metric SE \
  --questions_json lcb_release_v2_all_questions.json \
  --exp demo_run
```

Results are saved to:

```text
source/TTS/result/<exp_name>/result.jsonl
```

## Practical Notes

- `live_code_bench` downloads dataset content from Hugging Face via `datasets`.
- `embed`, `nlg_deberta`, and `nlg_llm` clustering may download large pretrained models the first time they run.
- `nlg_deberta` and `nlg_llm` are substantially heavier than `embed` or `bleu`.
- The test runner executes generated Python code. Use an isolated environment.
- Some defaults in the code assume internal paths such as `source/live_code_bench/result` and `source/ESE/result`, so running from the parent directory of `source/` is the safest option.
