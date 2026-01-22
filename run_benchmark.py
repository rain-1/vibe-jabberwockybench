#!/usr/bin/env python3
"""
Run the Jabberwocky benchmark across multiple models.

Usage:
    python run_benchmark.py --all                    # Run on all models
    python run_benchmark.py --claude                 # Run on Claude models only
    python run_benchmark.py --gpt                    # Run on GPT models only
    python run_benchmark.py --model claude-opus-4.5  # Run on specific model
    python run_benchmark.py --difficulty easy        # Run easy difficulty only
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from inspect_ai import eval
from inspect_ai.log import read_eval_log

from models_config import (
    MODELS,
    get_all_model_ids,
    get_claude_models,
    get_gpt_models,
    get_grok_models,
    get_gemini_models,
    get_model_id,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Jabberwocky benchmark on AI models via OpenRouter"
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--all",
        action="store_true",
        help="Run on all available models"
    )
    model_group.add_argument(
        "--claude",
        action="store_true",
        help="Run on all Claude models"
    )
    model_group.add_argument(
        "--gpt",
        action="store_true",
        help="Run on all GPT models"
    )
    model_group.add_argument(
        "--grok",
        action="store_true",
        help="Run on all Grok models"
    )
    model_group.add_argument(
        "--gemini",
        action="store_true",
        help="Run on all Gemini models"
    )
    model_group.add_argument(
        "--model",
        type=str,
        help="Run on a specific model (e.g., claude-opus-4.5)"
    )

    # Difficulty filter
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard", "extreme", "all"],
        default="all",
        help="Run only specific difficulty level (default: all)"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results (default: ./results)"
    )

    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Log individual sample results"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to run (for testing)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each model evaluation (default: 300)"
    )

    return parser.parse_args()


def get_models_to_run(args) -> list[str]:
    """Determine which models to run based on arguments."""
    if args.all:
        return get_all_model_ids()
    elif args.claude:
        return get_claude_models()
    elif args.gpt:
        return get_gpt_models()
    elif args.grok:
        return get_grok_models()
    elif args.gemini:
        return get_gemini_models()
    elif args.model:
        return [get_model_id(args.model)]
    else:
        raise ValueError("No model selection provided")


def get_task_name(difficulty: str) -> str:
    """Get the task function name based on difficulty."""
    if difficulty == "all":
        return "jabberwocky"
    else:
        return f"jabberwocky_{difficulty}"


def run_evaluation(model: str, task_name: str, args) -> dict:
    """Run evaluation for a single model."""
    print(f"\n{'='*60}")
    print(f"Running {task_name} on {model}")
    print(f"{'='*60}\n")

    try:
        # Set up evaluation parameters
        eval_params = {
            "tasks": f"jabberwocky_benchmark.py@{task_name}",
            "model": model,
            "log_dir": args.output_dir,
            "time_limit": args.timeout,
        }

        if args.max_samples:
            eval_params["limit"] = args.max_samples

        # Run evaluation
        logs = eval(**eval_params)

        # Extract results
        log = logs[0]
        results = {
            "model": model,
            "task": task_name,
            "timestamp": datetime.now().isoformat(),
            "status": log.status,
        }

        # Print summary
        print(f"\n{'='*60}")
        print(f"Results for {model}")
        print(f"{'='*60}")
        
        accuracy_score = "N/A"
        if log.results and log.results.scores:
            for score in log.results.scores:
                # In inspect-ai 0.3+, scores is a list of EvalScore
                # Metrics are inside score.metrics
                if 'accuracy' in score.metrics:
                    val = score.metrics['accuracy'].value
                    accuracy_score = f"{val:.2%}"
                    print(f"Accuracy: {accuracy_score}")
                
                for metric_name, metric_val in score.metrics.items():
                    if metric_name != 'accuracy':
                        print(f"{metric_name}: {metric_val.value:.4f}")
        print()

        # Update results dict for the summary file
        results["accuracy"] = accuracy_score
        return results

    except Exception as e:
        print(f"ERROR running {model}: {str(e)}", file=sys.stderr)
        return {
            "model": model,
            "task": task_name,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }


def main():
    """Main execution function."""
    args = parse_args()

    # Check for OpenRouter API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set!", file=sys.stderr)
        print("Please set it with: export OPENROUTER_API_KEY='your-key-here'", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get models to run
    models = get_models_to_run(args)
    task_name = get_task_name(args.difficulty)

    print(f"\n{'='*60}")
    print(f"Jabberwocky Benchmark")
    print(f"{'='*60}")
    print(f"Task: {task_name}")
    print(f"Models: {len(models)}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Run evaluations
    all_results = []
    for model in models:
        result = run_evaluation(model, task_name, args)
        all_results.append(result)

    # Save aggregated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"summary_{timestamp}.json"

    with open(summary_file, 'w') as f:
        json.dump({
            "benchmark": "jabberwocky",
            "task": task_name,
            "timestamp": timestamp,
            "models_evaluated": len(models),
            "results": all_results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Benchmark complete!")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*60}\n")

    # Print comparison table
    print("\nComparison Summary:")
    print(f"{'Model':<40} {'Accuracy':<10} {'Status':<10}")
    print("-" * 60)
    for result in all_results:
        model_name = result['model'].split('/')[-1][:38]
        accuracy = result.get('accuracy', 'N/A')
        status = result.get('status', 'unknown')
        print(f"{model_name:<40} {accuracy:<10} {status:<10}")


if __name__ == "__main__":
    main()
