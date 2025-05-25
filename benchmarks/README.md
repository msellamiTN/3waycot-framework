# 3WayCoT Benchmarking Framework

This benchmarking framework allows you to evaluate the 3WayCoT approach against other reasoning methodologies such as standard Chain of Thought (CoT), Tree of Thought (ToT), and Graph of Thought (GoT).

## Features

- **Uncertainty-focused evaluation**: Specifically designed to assess how well different reasoning approaches handle uncertainty, ambiguity, and incomplete information
- **Comparative analysis**: Evaluate 3WayCoT against standard CoT approaches
- **Detailed metrics**: Track uncertainty acknowledgment, assumption coverage, decision distribution, and more
- **Multiple LLM providers**: Support for Gemini, OpenAI, Anthropic, and local models

## Benchmark Datasets

### Uncertainty Benchmark

Located at `datasets/uncertainty_benchmark.json`, this dataset includes problems that inherently involve uncertainty and require reasoning about ambiguous or incomplete information. Each problem includes:

- **Prompt and context**: The question and relevant background information
- **Expected assumptions**: Key assumptions that should be made explicit in high-quality reasoning
- **Uncertainty factors**: Specific areas of uncertainty that should be acknowledged
- **Gold answer**: Reference solution that appropriately handles the uncertainty
- **Reasoning quality metrics**: Target metrics for high-quality reasoning on this problem

## Running Benchmarks

### Basic Usage

```bash
python run_uncertainty_benchmark.py --provider gemini
```

This will run the 3WayCoT framework on the uncertainty benchmark using the Gemini provider.

### Comparative Evaluation

```bash
python run_uncertainty_benchmark.py --provider gemini --compare
```

This will run both 3WayCoT and standard CoT on the benchmark and compare their performance.

### Advanced Options

```bash
python run_uncertainty_benchmark.py --provider gemini --model gemini-1.5-flash --alpha 0.75 --beta 0.5 --tau 0.45 --max-items 2 --compare
```

- `--provider`: LLM provider (gemini, openai, anthropic, local)
- `--model`: Specific model to use (optional)
- `--alpha`: Acceptance threshold for three-way decisions (default: 0.7)
- `--beta`: Rejection threshold for three-way decisions (default: 0.6)
- `--tau`: Threshold for fuzzy membership in TFCA (default: 0.4)
- `--max-items`: Maximum number of items to test (for quick testing)
- `--compare`: Compare 3WayCoT with standard CoT
- `--output`: Path to save benchmark results (default: benchmark_results.json)

## Interpreting Results

The benchmark produces detailed results as a JSON file with the following key metrics:

- **Uncertainty acknowledgment score**: How well the reasoning acknowledges the identified uncertainty factors
- **Assumption coverage**: Percentage of expected assumptions that were explicitly stated
- **Defer ratio**: Percentage of reasoning steps where the model acknowledged uncertainty (DEFER decision)
- **Decision distribution**: Counts of ACCEPT, REJECT, and DEFER decisions

For comparative evaluations, additional metrics include:

- **Relative time increase**: How much longer 3WayCoT takes compared to standard CoT
- **Average processing times**: For both approaches

## Creating Custom Benchmarks

To create a custom benchmark, follow the structure in `datasets/uncertainty_benchmark.json`. The key elements are:

1. A list of benchmark items with prompts and contexts
2. Expected assumptions for each item
3. Uncertainty factors that should be acknowledged
4. Gold answers that represent ideal reasoning under uncertainty

Custom benchmarks should be placed in the `datasets/` directory.

## Example Output

```json
{
  "benchmark_info": {
    "dataset": "uncertainty_benchmark.json",
    "provider": "gemini",
    "model": "gemini-1.5-flash",
    "num_items": 5,
    "timestamp": "2025-05-20 16:45:12"
  },
  "overall_metrics": {
    "average_uncertainty_acknowledgment": 0.7234,
    "average_assumption_coverage": 0.6851,
    "average_defer_ratio": 0.2143
  }
}
```
