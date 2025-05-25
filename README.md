# 3WayCoT: Three-Way Chain of Thought Framework

A sophisticated framework for AI reasoning that combines Chain of Thought (CoT) with Three-Way Decisions and Triadic Fuzzy Concept Analysis.

## Overview

The 3WayCoT framework enhances LLM reasoning by:

1. Generating detailed step-by-step reasoning chains with explicit assumptions
2. Evaluating confidence for each reasoning step
3. Making three-way decisions (Accept/Reject/Abstain) based on confidence thresholds
4. Providing rich visualizations of confidence metrics and decision processes
5. Supporting parametrized decision making with customizable thresholds

## New Features

- **Parametrized Gradio Interface**: Interactive UI with customizable parameters
- **Enhanced Confidence Metrics**: Improved extraction and prioritization of confidence values
- **Final Answer Extraction**: Automatically identifies and extracts answers from each reasoning step
- **Decision Visualization**: Rich visualizations of confidence metrics and decision processes
- **Model Selection**: Choose from multiple LLM providers and models
- **Use Case Templates**: Predefined parameter sets for different reasoning scenarios

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your API keys in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Quick Start Guide

### Parametrized Gradio Interface

The recommended way to use the framework is through the enhanced parametrized Gradio interface:

```bash
python launch_parametrized_app.py
```

This will start the interface on http://localhost:7860

### Command Line Usage

You can also use the framework from the command line:

```bash
python main.py --prompt "Your prompt here" --provider gemini
```

### API Server

For integration with other applications, you can use the FastAPI server:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Using the Parametrized Interface

### Parameter Controls

- **Decision Parameters**:
  - **Alpha Threshold** (0.5-0.9): Higher values lead to more conservative decisions
  - **Beta Threshold** (0.3-0.7): Lower values increase rejection rate
  - **Tau Threshold** (0.3-0.8): Confidence value threshold
  - **Confidence Weight** (0.3-1.0): Weight given to confidence in the decision-making process

- **Model Parameters**:
  - **Model**: Select from available language models
  - **Temperature** (0.1-1.0): Controls randomness of model output
  - **Max Tokens** (500-4000): Maximum output length
  - **Top P** (0.1-1.0): Controls diversity of model output

### Use Case Selection

The interface provides predefined use cases with optimized parameters for different scenarios:

- **Default**: Balanced parameters for general use
- **Conservative**: Higher thresholds for more cautious decisions
- **Exploratory**: Lower thresholds for more inclusive reasoning

### Visualizations

- **Confidence Analysis**: Distribution and patterns of confidence across reasoning steps
- **Decision Analysis**: Balance between accept/reject/abstain decisions
- **Triadic Lattice**: 3D visualization of the concept space
- **Reasoning Steps**: Detailed breakdown of each step with metrics
- **Raw Output**: Complete JSON for detailed examination

## Framework Architecture

### Core Components

1. **Chain of Thought Generator**: Produces detailed reasoning steps with confidence values
2. **Three-Way Decision Maker**: Evaluates reasoning with Accept/Reject/Abstain decisions
3. **Triadic FCA Engine**: Analyzes concept relationships in three dimensions
4. **Uncertainty Resolver**: Handles uncertain or conflicting reasoning

### Confidence Metrics

The framework implements enhanced confidence metrics:

- **Explicit Confidence**: Values directly stated in reasoning
- **Membership Degrees**: Fuzzy membership in decision categories
- **Decision Consistency**: Agreement between different reasoning steps
- **Weighted Confidence**: Prioritized confidence values based on context

## Configuration

The framework is configured through `config.yml` with settings for:

- Decision thresholds (alpha, beta, tau)
- LLM providers and models
- Default parameters
- Logging and output formats
- Use case definitions

## Examples

### Medical Reasoning

```python
prompt = "A 39-year-old woman with fever and left lower quadrant pain. Lab shows platelet count 14,200/mm3. What is the likely diagnosis?"
result = app.process(prompt, alpha=0.8, beta=0.5, tau=0.7)
```

### Legal Analysis

```python
prompt = "Analyze the legal implications of a company using customer data for AI training without explicit consent."
result = app.process(prompt, alpha=0.7, beta=0.4, tau=0.6)
```

## Advanced Usage

### Dataset Processing

The interface supports batch processing of datasets:

1. Place JSON files in the `data/benchmarks` directory
2. Select the dataset from the dropdown in the interface
3. Click "Run Dataset" to process all items

### Custom Use Cases

Define your own use cases in `config.yml`:

```yaml
use_cases:
  my_custom_case:
    alpha: 0.75
    beta: 0.45
    tau: 0.65
    description: "Custom parameters for specific domain"
```

## Acknowledgments

This framework builds on research in:
- Chain of Thought Reasoning
- Three-Way Decision Theory
- Triadic Fuzzy Concept Analysis
- Explainable AI techniques

## License

This project is licensed under the MIT License - see the LICENSE file for details.
