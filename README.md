# Jabberwocky Benchmark

A benchmark for evaluating AI models' ability to interpret "Jabberwocky" language - sentences where content words are replaced with nonsense strings while preserving grammatical structure.

## Overview

This benchmark is based on the research paper ["The unreasonable effectiveness of pattern matching"](https://arxiv.org/abs/2601.11432) by Gary Lupyan and Blaise Agüera y Arcas. The paper demonstrates that large language models can make sense of Jabberwocky language through pattern matching and structural understanding.

### Example

**Jabberwocky Input:** "He dwushed a ghanc zawk"

**Expected Translation:** "He dragged a spare chair"

The model must use grammatical structure, context clues, and pattern matching to recover meaning from nonsense words.

## Features

- **25 carefully designed test cases** across three difficulty levels (easy, medium, hard)
- **Multiple linguistic categories**: simple actions, descriptions, questions, comparisons, conditional statements, and more
- **Comprehensive model coverage** via OpenRouter:
  - Claude (Opus 4.5, Sonnet 4.5, Haiku 4)
  - GPT (ChatGPT 5.2, GPT-4.5 Turbo)
  - Grok (Grok 2, Grok 2 Vision)
  - Gemini (2.0 Flash, 2.0 Pro, Pro)
- **Built on Inspect AI framework** - industry-standard evaluation framework from UK AISI
- **Semantic scoring** with partial credit for close matches

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenRouter API key ([get one here](https://openrouter.ai/keys))

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd vibe-jabberwockybench
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenRouter API key:
```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
export OPENROUTER_API_KEY='your-key-here'
```

## Usage

### Quick Start

Run the benchmark on all models:
```bash
python run_benchmark.py --all
```

### Run on Specific Model Families

```bash
# Test only Claude models
python run_benchmark.py --claude

# Test only GPT models
python run_benchmark.py --gpt

# Test only Grok models
python run_benchmark.py --grok

# Test only Gemini models
python run_benchmark.py --gemini
```

### Run on Specific Model

```bash
python run_benchmark.py --model claude-opus-4.5
```

### Filter by Difficulty

```bash
# Run only easy difficulty tests
python run_benchmark.py --all --difficulty easy

# Run only hard difficulty tests
python run_benchmark.py --claude --difficulty hard
```

### Additional Options

```bash
# Limit samples for quick testing
python run_benchmark.py --model claude-opus-4.5 --max-samples 5

# Specify custom output directory
python run_benchmark.py --all --output-dir ./my_results

# Enable detailed sample logging
python run_benchmark.py --all --log-samples
```

## Using Inspect AI Directly

You can also use the Inspect AI CLI directly:

```bash
# Run the full benchmark
inspect eval jabberwocky_benchmark.py@jabberwocky --model openrouter/anthropic/claude-opus-4.5

# Run specific difficulty
inspect eval jabberwocky_benchmark.py@jabberwocky_easy --model openrouter/anthropic/claude-sonnet-4.5

# View results
inspect view
```

## Dataset Structure

The benchmark includes 25 test cases organized as follows:

### Difficulty Levels

- **Easy** (8 cases): Simple sentence structures with basic verb-object patterns
- **Medium** (9 cases): More complex structures with temporal sequences, comparisons, and conditional statements
- **Hard** (8 cases): Advanced structures including counterfactuals, simultaneous actions, and abstract concepts

### Linguistic Categories

- Simple actions and descriptions
- Questions (direct and comparative)
- Temporal and causal sequences
- Comparisons and superlatives
- Conditional and counterfactual statements
- Negation and disjunction
- Progressive and perfect tenses
- Formal and abstract language

## Scoring

The benchmark uses a custom semantic similarity scorer:

- **Correct (C)**: Exact match or high semantic similarity (≥70% word overlap)
- **Partial (P)**: Moderate semantic similarity (40-70% word overlap)
- **Incorrect (I)**: Low semantic similarity (<40% word overlap)

Final accuracy is calculated based on correct answers, with additional metrics for partial credit.

## Results

Results are saved to the `results/` directory:

- Individual evaluation logs (JSON format compatible with Inspect AI)
- Aggregated summary with comparison across models
- Timestamps and metadata for reproducibility

## Model Configuration

All model configurations are defined in `models_config.py`. The benchmark is configured for:

### Claude Models
- `claude-opus-4.5` - Most capable Claude model
- `claude-sonnet-4.5` - Balanced performance
- `claude-haiku-4` - Fast and efficient

### GPT Models
- `chatgpt-5.2` - Latest ChatGPT
- `gpt-4.5-turbo` - GPT 4.5 Turbo
- `gpt-4-turbo` - Previous generation

### Grok Models
- `grok-2` - xAI's model
- `grok-2-vision` - With vision capabilities

### Gemini Models
- `gemini-2.0-pro` - Most capable Gemini
- `gemini-2.0-flash` - Fast and efficient
- `gemini-pro` - Previous generation

## Architecture

### Files

```
.
├── jabberwocky_benchmark.py   # Main benchmark implementation (Inspect tasks)
├── jabberwocky_dataset.json   # Test cases dataset
├── models_config.py           # Model configurations for OpenRouter
├── run_benchmark.py           # Convenience script for running evaluations
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variable template
└── README.md                 # This file
```

### Key Components

1. **Dataset** (`jabberwocky_dataset.json`): JSON file with test cases
2. **Tasks** (`jabberwocky_benchmark.py`): Inspect AI task definitions with custom scorers
3. **Models** (`models_config.py`): OpenRouter model configurations
4. **Runner** (`run_benchmark.py`): CLI for batch evaluations

## Extending the Benchmark

### Adding New Test Cases

Edit `jabberwocky_dataset.json`:

```json
{
  "id": "jab_026",
  "input": "Your jabberwocky sentence here",
  "target": "The expected translation",
  "difficulty": "medium",
  "category": "your_category"
}
```

### Adding New Models

Add to `models_config.py`:

```python
"your-model-key": {
    "name": "provider/model-name",
    "provider": "openrouter",
    "description": "Model description"
}
```

### Custom Scorers

Modify the `jabberwocky_scorer()` function in `jabberwocky_benchmark.py` to implement different scoring logic.

## Research Context

This benchmark evaluates the capability described in the paper ["The unreasonable effectiveness of pattern matching"](https://arxiv.org/abs/2601.11432), which demonstrates that:

1. LLMs can interpret nonsense language using structural patterns
2. Pattern matching is not separate from "real" intelligence but a key component
3. Grammatical structure provides sufficient signal for meaning recovery

## License

MIT License

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{lupyan2026unreasonable,
  title={The unreasonable effectiveness of pattern matching},
  author={Lupyan, Gary and Agüera y Arcas, Blaise},
  journal={arXiv preprint arXiv:2601.11432},
  year={2026}
}
```

## Contributing

Contributions are welcome! Please feel free to:

- Add more test cases
- Improve scoring algorithms
- Add support for more models
- Report issues or suggest improvements

## Sources

- [The unreasonable effectiveness of pattern matching (arXiv)](https://arxiv.org/abs/2601.11432)
- [Inspect AI Framework](https://github.com/UKGovernmentBEIS/inspect_ai)
- [Inspect AI Documentation](https://inspect.aisi.org.uk/)
- [OpenRouter API](https://openrouter.ai/)
