# Ensemble Semantic Entropy

This repository provides a workflow for code generation on LiveCodeBench, entropy-based clustering and evaluation, and test-time scaling with both baseline and cascade strategies.

## Modules

- `live_code_bench`: dataset loading, prompting, generation, self-debugging, and test execution. The main entry point is `live_code_bench/interface.py`.
- `ESE`: semantic-entropy computation and uncertainty estimation. It includes clustering methods, entropy formulas, and evaluation scripts.
- `TTS`: selection and reranking components built on top of saved generation results, including baseline methods and a multi-layer cascade pipeline.

## Configuration

### Model configuration

Create a JSON file for generation, for example `live_code_bench/config.json`:

```json
{
  "generate_lm": {
    "model_name": "openai/Qwen3-8B",
    "api_base": "http://localhost:8001/v1",
    "api_key": "your-api-key",
    "timeout": 120,
    "temperature": 0.7,
    "max_tokens": 8192,
    "cache": true
  }
}
```

The `generate_lm` section is required by the generation pipeline. Additional DSPy or LiteLLM-compatible fields can be added if needed.

### TTS layer configuration

Create a JSON file for baseline and cascade selection, for example `TTS/configs/config.json`:

```json
{
  "layers_count": 2,
  "layers": [
    {
      "layer_idx": 0,
      "samples": {
        "qwen3_8b": [0, 1, 2, 3, 4]
      },
      "alpha": 1.0,
      "beta": 1.0,
      "threshold": 0.5
    },
    {
      "layer_idx": 1,
      "samples": {
        "qwen3_8b": [5, 6, 7, 8, 9]
      },
      "alpha": 1.0,
      "beta": 1.0,
      "threshold": 0.2
    }
  ]
}
```

In this file, `samples` maps each saved model result directory to the round indices used by that layer. `baselines.py` reads the first layer, while `cascade_tts.py` uses the full layer list.

### Question metadata

Downstream evaluation expects a question metadata JSON file such as `lcb_release_v2_all_questions.json`. Each record should include at least `task_id`, `prompt`, and `is_stdin`.

## How to Run

Install dependencies from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the workflow in three stages:

1. Generate LiveCodeBench runs.

```bash
python3 -m live_code_bench.interface \
  --model_name qwen3_8b \
  --full \
  --config_path live_code_bench/config.json \
  --dataset_difficulty easy \
  --num_rounds 3 \
  --num_workers 4
```

This produces per-round generation outputs under `live_code_bench/result/<difficulty>/<model_name>/`.

2. Compute semantic entropy and clustering results.

```bash
python3 -m ESE.calc_entropy \
  --model qwen3_8b \
  --difficulty easy \
  --method embed \
  --rounds 0,5,10,15,19 \
  --questions-json lcb_release_v2_all_questions.json \
  --result-dir live_code_bench/result \
  --output-dir ESE/result
```

3. Run evaluation or selection.

Entropy evaluation:

```bash
python3 -m ESE.evaluate \
  --result-dir ESE/result \
  --output-dir ESE/evaluation
```

Baseline:

```bash
python3 -m TTS.baselines \
  --config TTS/configs/config.json \
  --result_base live_code_bench/result \
  --difficulty easy \
  --method major_voting \
  --clustering_method embed \
  --questions_json lcb_release_v2_all_questions.json
```

Cas:

```bash
python3 -m TTS.cascade_tts \
  --config TTS/configs/config.json \
  --result_base live_code_bench/result \
  --difficulty easy \
  --clustering_method embed \
  --metric SE \
  --questions_json lcb_release_v2_all_questions.json \
  --exp demo_run
```
