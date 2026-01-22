#!/usr/bin/env python3
"""
Test script to verify the benchmark setup is correct.

Run this after installation to check that everything is configured properly.
"""

import json
import os
import sys
from pathlib import Path


def check_python_version():
    """Check Python version is 3.10 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import inspect_ai
        print(f"✅ inspect-ai installed (version: {inspect_ai.__version__})")
        return True
    except ImportError:
        print("❌ inspect-ai not installed. Run: pip install -r requirements.txt")
        return False


def check_dataset():
    """Check if dataset file exists and is valid."""
    dataset_path = Path("jabberwocky_dataset.json")
    if not dataset_path.exists():
        print("❌ Dataset file not found: jabberwocky_dataset.json")
        return False

    try:
        with open(dataset_path) as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("❌ Dataset should be a list")
            return False

        total = len(data)
        easy = sum(1 for d in data if d.get("difficulty") == "easy")
        medium = sum(1 for d in data if d.get("difficulty") == "medium")
        hard = sum(1 for d in data if d.get("difficulty") == "hard")

        print(f"✅ Dataset valid: {total} test cases")
        print(f"   - Easy: {easy}")
        print(f"   - Medium: {medium}")
        print(f"   - Hard: {hard}")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ Dataset JSON invalid: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False


def check_modules():
    """Check if benchmark modules can be imported."""
    try:
        import jabberwocky_benchmark
        print("✅ jabberwocky_benchmark module loads successfully")
    except Exception as e:
        print(f"❌ Error importing jabberwocky_benchmark: {e}")
        return False

    try:
        import models_config
        total_models = len(models_config.MODELS)
        print(f"✅ models_config module loads successfully ({total_models} models)")
    except Exception as e:
        print(f"❌ Error importing models_config: {e}")
        return False

    return True


def check_api_key():
    """Check if OpenRouter API key is set."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        print("⚠️  OPENROUTER_API_KEY not set")
        print("   Set it with: export OPENROUTER_API_KEY='your-key-here'")
        return False
    print("✅ OPENROUTER_API_KEY is set")
    return True


def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("Jabberwocky Benchmark - Setup Verification")
    print("="*60 + "\n")

    checks = [
        check_python_version(),
        check_dependencies(),
        check_dataset(),
        check_modules(),
        check_api_key(),
    ]

    print("\n" + "="*60)
    if all(checks[:-1]):  # All except API key (which is just a warning)
        print("✅ Setup is complete and valid!")
        if not checks[-1]:
            print("\n⚠️  Note: Set OPENROUTER_API_KEY to run actual benchmarks")
        print("\nYou can now run:")
        print("  python run_benchmark.py --model claude-opus-4.5 --max-samples 2")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        sys.exit(1)
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
