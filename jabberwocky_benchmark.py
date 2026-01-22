"""
Jabberwocky Benchmark - Pattern Matching Evaluation

Based on the research paper "The unreasonable effectiveness of pattern matching"
by Gary Lupyan and Blaise Agüera y Arcas (arXiv:2601.11432)

This benchmark evaluates LLMs' ability to make sense of "Jabberwocky" language
where content words are replaced with nonsense strings, testing pattern matching
and structural understanding capabilities.
"""

import json
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import (
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import (
    Generate,
    TaskState,
    generate,
    solver,
)


# Custom prompt that explains the task
JABBERWOCKY_SYSTEM_PROMPT = """You are participating in a linguistic evaluation. You will be given sentences where content words (nouns, verbs, adjectives, adverbs) have been replaced with nonsense words, but the grammatical structure remains intact.

Your task is to interpret what the original sentence might have meant by using context clues, grammatical structure, and pattern matching.

Provide ONLY the interpreted sentence as your answer. Do not provide explanations or multiple options."""


@solver
def jabberwocky_prompt() -> solver:
    """Add task-specific instructions to the prompt."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Add system message with instructions
        state.messages.insert(
            0,
            ChatMessageUser(content=JABBERWOCKY_SYSTEM_PROMPT)
        )

        # Format the user prompt
        jabberwocky_sentence = state.input_text
        state.messages.append(
            ChatMessageUser(
                content=f"Translate this Jabberwocky sentence to normal English:\n\n\"{jabberwocky_sentence}\""
            )
        )

        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def jabberwocky_scorer() -> scorer:
    """
    Custom scorer for Jabberwocky translations.

    Uses fuzzy matching to evaluate semantic similarity between
    the model's output and the target translation.
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Get the model's answer
        model_answer = state.output.completion.strip().lower()

        # Get the target answer(s)
        target_text = target.text.strip().lower() if isinstance(target.text, str) else target.text[0].strip().lower()

        # Exact match scoring
        if model_answer == target_text:
            return Score(
                value="C",  # Correct
                answer=model_answer,
                explanation="Exact match with target translation"
            )

        # Partial credit for semantic similarity
        # Check if key content words match
        model_words = set(model_answer.split())
        target_words = set(target_text.split())

        # Remove common stop words for better matching
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'was', 'were', 'is', 'are', 'be', 'been', 'being'}
        model_words_filtered = model_words - stop_words
        target_words_filtered = target_words - stop_words

        if len(target_words_filtered) == 0:
            # Fallback to unfiltered comparison
            overlap = len(model_words & target_words)
            total = len(target_words)
        else:
            overlap = len(model_words_filtered & target_words_filtered)
            total = len(target_words_filtered)

        similarity = overlap / total if total > 0 else 0

        # Scoring thresholds
        if similarity >= 0.7:
            return Score(
                value="C",
                answer=model_answer,
                explanation=f"High semantic similarity ({similarity:.2%})"
            )
        elif similarity >= 0.4:
            return Score(
                value="P",  # Partial
                answer=model_answer,
                explanation=f"Moderate semantic similarity ({similarity:.2%})"
            )
        else:
            return Score(
                value="I",  # Incorrect
                answer=model_answer,
                explanation=f"Low semantic similarity ({similarity:.2%})"
            )

    return score


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert a dataset record to an Inspect Sample."""
    return Sample(
        input=record["input"],
        target=record["target"],
        id=record.get("id"),
        metadata={
            "difficulty": record.get("difficulty", "unknown"),
            "category": record.get("category", "unknown")
        }
    )


@task
def jabberwocky():
    """
    Jabberwocky Pattern Matching Benchmark

    Evaluates LLM ability to interpret nonsense language using
    grammatical structure and pattern matching.
    """
    return Task(
        dataset=json_dataset(
            "jabberwocky_dataset.json",
            sample_fields=record_to_sample
        ),
        solver=[
            jabberwocky_prompt(),
            generate()
        ],
        scorer=jabberwocky_scorer(),
    )


@task
def jabberwocky_easy():
    """Jabberwocky benchmark - Easy difficulty only"""
    dataset = json_dataset(
        "jabberwocky_dataset.json",
        sample_fields=record_to_sample
    )

    # Filter for easy samples
    def filter_easy(sample: Sample) -> bool:
        return sample.metadata.get("difficulty") == "easy"

    return Task(
        dataset=[s for s in dataset if filter_easy(s)],
        solver=[jabberwocky_prompt(), generate()],
        scorer=jabberwocky_scorer(),
    )


@task
def jabberwocky_medium():
    """Jabberwocky benchmark - Medium difficulty only"""
    dataset = json_dataset(
        "jabberwocky_dataset.json",
        sample_fields=record_to_sample
    )

    def filter_medium(sample: Sample) -> bool:
        return sample.metadata.get("difficulty") == "medium"

    return Task(
        dataset=[s for s in dataset if filter_medium(s)],
        solver=[jabberwocky_prompt(), generate()],
        scorer=jabberwocky_scorer(),
    )


@task
def jabberwocky_hard():
    """Jabberwocky benchmark - Hard difficulty only"""
    dataset = json_dataset(
        "jabberwocky_dataset.json",
        sample_fields=record_to_sample
    )

    def filter_hard(sample: Sample) -> bool:
        return sample.metadata.get("difficulty") == "hard"

    return Task(
        dataset=[s for s in dataset if filter_hard(s)],
        solver=[jabberwocky_prompt(), generate()],
        scorer=jabberwocky_scorer(),
    )
