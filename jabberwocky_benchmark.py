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


def word_levenshtein_distance(s1: list[str], s2: list[str]) -> int:
    """
    Compute Levenshtein distance at the word level.
    Returns the minimum number of word insertions, deletions, 
    and substitutions needed to transform s1 into s2.
    """
    m, n = len(s1), len(s2)
    
    # Create distance matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )
    
    return dp[m][n]


def word_level_similarity(model_text: str, target_text: str) -> float:
    """
    Compute similarity between two texts using word-level Levenshtein distance.
    Returns a score between 0.0 (completely different) and 1.0 (identical).
    """
    # Tokenize into words, normalizing case and removing punctuation
    import re
    
    def tokenize(text: str) -> list[str]:
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    model_words = tokenize(model_text)
    target_words = tokenize(target_text)
    
    if not model_words and not target_words:
        return 1.0  # Both empty = perfect match
    if not model_words or not target_words:
        return 0.0  # One empty, one not = no match
    
    distance = word_levenshtein_distance(model_words, target_words)
    max_len = max(len(model_words), len(target_words))
    
    # Convert distance to similarity (0 distance = 1.0 similarity)
    similarity = 1.0 - (distance / max_len)
    return max(0.0, similarity)  # Clamp to non-negative


@scorer(metrics=[accuracy(), stderr()])
def jabberwocky_scorer() -> scorer:
    """
    Custom scorer for Jabberwocky translations.

    Uses word-level Levenshtein distance to evaluate similarity between
    the model's output and the target translation. This gives a more
    nuanced score than simple word overlap.
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Get the model's answer
        model_answer = state.output.completion.strip()

        # Get the target answer(s)
        target_text = target.text.strip() if isinstance(target.text, str) else target.text[0].strip()

        # Compute word-level similarity
        similarity = word_level_similarity(model_answer, target_text)

        # Scoring thresholds based on edit distance similarity
        if similarity >= 0.9:
            return Score(
                value="C",  # Correct - very close match
                answer=model_answer,
                explanation=f"Excellent match (similarity: {similarity:.1%})"
            )
        elif similarity >= 0.7:
            return Score(
                value="C",  # Correct - good match
                answer=model_answer,
                explanation=f"Good match (similarity: {similarity:.1%})"
            )
        elif similarity >= 0.5:
            return Score(
                value="P",  # Partial - moderate match
                answer=model_answer,
                explanation=f"Partial match (similarity: {similarity:.1%})"
            )
        elif similarity >= 0.3:
            return Score(
                value="P",  # Partial - some overlap
                answer=model_answer,
                explanation=f"Weak match (similarity: {similarity:.1%})"
            )
        else:
            return Score(
                value="I",  # Incorrect
                answer=model_answer,
                explanation=f"Poor match (similarity: {similarity:.1%})"
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
            "category": record.get("category", "unknown"),
            "source": record.get("source", "unknown")
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
def jabberwocky_extreme():
    """Jabberwocky benchmark - Extreme difficulty only (BLANK versions)"""
    dataset = json_dataset(
        "jabberwocky_dataset.json",
        sample_fields=record_to_sample
    )

    def filter_extreme(sample: Sample) -> bool:
        return sample.metadata.get("difficulty") == "extreme"

    return Task(
        dataset=[s for s in dataset if filter_extreme(s)],
        solver=[jabberwocky_prompt(), generate()],
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
