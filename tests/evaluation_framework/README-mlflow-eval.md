# MLflow Prompt and LLM Evaluation

This directory contains an MLflow-based evaluation system for comparing different prompts and LLM models for the PromptParser task.

## Overview

The `mlflow_prompt_eval.py` script provides a comprehensive way to:

- Test multiple LLM models against the same dataset
- Compare different system prompts
- Track all results with MLflow experiment tracking
- Generate detailed reports and comparisons
- Log metrics, parameters, and artifacts

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements-mlflow.txt
```

### 2. Start MLflow Server (Optional)

If you want to use the MLflow UI for tracking:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Then access the UI at `http://localhost:5000`

### 3. Ensure Ollama is Running

Make sure your Ollama server is running at the specified base URL (default: `http://localhost:11434`).

## Usage

### Basic Usage

Run evaluation with default settings:

```bash
python mlflow_prompt_eval.py
```

### Advanced Usage

#### Custom Models and Prompts

```bash
python mlflow_prompt_eval.py \
  --models "hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL" \
         "hf.co/unsloth/Phi-4-mini-reasoning-GGUF:BF16" \
  --system-prompts "default" "strict_json" "conversational"
```

#### Custom MLflow Tracking

```bash
python mlflow_prompt_eval.py \
  --tracking-uri "http://localhost:5000" \
  --experiment-name "custom_prompt_evaluation"
```

#### Performance Options

```bash
python mlflow_prompt_eval.py \
  --no-fair-latency \
  --cooldown-ms 50
```

#### Compare Existing Runs Only

```bash
python mlflow_prompt_eval.py --compare-only
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--experiment-name` | Name of the MLflow experiment | `prompt_parser_evaluation` |
| `--models` | List of models to evaluate | Mistral Small 3.2 |
| `--system-prompts` | List of system prompt keys | All available prompts |
| `--base-url` | Base URL for Ollama API | `http://localhost:11434` |
| `--no-fair-latency` | Disable fair latency cooldown | False |
| `--cooldown-ms` | Cooldown time between runs (ms) | 150 |
| `--compare-only` | Only compare existing runs | False |
| `--tracking-uri` | MLflow tracking URI | Default MLflow URI |

## Available System Prompts

The script includes several system prompt variants:

- `default` - Comprehensive expert prompt with detailed instructions
- `strict_json` - Minimal prompt focused on exact JSON output
- `conversational` - Friendly, natural language prompt
- `step_by_step` - Analytical, step-by-step approach
- `minimal_1`, `minimal_2`, `minimal_3` - Various minimal prompt variations

## Metrics Tracked

For each model-prompt combination, the script tracks:

### Performance Metrics
- **Pass Rate**: Percentage of test cases passed
- **Total Cases**: Number of test cases evaluated
- **Passed/Failed Count**: Raw counts of passed and failed cases

### Latency Metrics
- **Average Latency**: Mean response time in milliseconds
- **P50 Latency**: Median response time (50th percentile)
- **P95 Latency**: 95th percentile response time

### Detailed Results
- Per-case pass/fail status
- Per-case latency
- Detailed failure reasons
- Actual vs expected outputs

## MLflow Artifacts

The script logs several artifacts to MLflow:

1. **System Prompts**: Text files of each system prompt used
2. **Results**: CSV files with detailed results for each run
3. **Failures**: Text files listing all failure cases
4. **Summary**: Overall summary CSV with aggregated metrics

## Output Files

In addition to MLflow tracking, the script saves local files:

- `/tmp/full_results.csv` - Complete results dataset
- `/tmp/final_comparison.csv` - Comparison table of all runs
- `/tmp/run_comparison.csv` - Comparison when using `--compare-only`

## Example Workflow

### 1. Initial Evaluation

```bash
python mlflow_prompt_eval.py \
  --models "hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL" \
  --system-prompts "default" "strict_json"
```

### 2. View Results in MLflow UI

Navigate to `http://localhost:5000` and explore:
- Experiment overview
- Individual run details
- Parameter and metric comparisons
- Artifact downloads

### 3. Compare Performance

```bash
python mlflow_prompt_eval.py --compare-only
```

This will show a ranked comparison of all runs by pass rate and latency.

## Integration with Existing Tests

The MLflow evaluator uses the same test data and evaluation logic as the original unittest:

- **Test Data**: Uses `tests/test_data.py` (DATASET and SYSTEM_PROMPTS)
- **Evaluation Logic**: Uses `tests/test_utils.py` for validation
- **PromptParser**: Same implementation as the main codebase

This ensures consistency between unit tests and MLflow experiments.

## Customization

### Adding New Models

Simply add the model identifier to the `--models` argument:

```bash
python mlflow_prompt_eval.py \
  --models "your-new-model-name" "existing-model-name"
```

### Adding New System Prompts

1. Add your prompt to `SYSTEM_PROMPTS` in `tests/test_data.py`
2. Include the key in the `--system-prompts` argument

### Custom Metrics

The script can be extended to log additional metrics by modifying the `_log_overall_summary` method.

## Troubleshooting

### Common Issues

1. **Ollama Connection**: Ensure Ollama is running and accessible at the specified URL
2. **Model Availability**: Make sure the specified models are pulled in Ollama
3. **Memory Issues**: Use `--no-fair-latency` if experiencing memory-related timeouts
4. **MLflow Server**: Start the MLflow server if you want to use the UI

### Debug Mode

For debugging, you can run with a single model and prompt:

```bash
python mlflow_prompt_eval.py \
  --models "hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL" \
  --system-prompts "default"
```

## Performance Considerations

- **Fair Latency**: The default cooldown ensures fair comparison but increases total runtime
- **Parallel Execution**: Consider running multiple experiments with different model subsets
- **Resource Usage**: Monitor GPU/CPU usage when running multiple large models

## Example Output

```
=== Final Comparison ===
                                              model  system_prompt  pass_rate  latency_avg_ms  latency_p50_ms  latency_p95_ms
0  hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL  strict_json      0.923          1250.3          1180.5          1450.7
1  hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL        default      0.897          1320.1          1250.8          1520.3
2  hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL  conversational      0.885          1180.7          1120.2          1380.9
```

This output shows the ranking of model-prompt combinations by pass rate and latency, helping you identify the best performing configuration.
