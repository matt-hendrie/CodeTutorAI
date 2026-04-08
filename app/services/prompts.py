"""Prompt templates and builder for LLM question generation."""

from __future__ import annotations

from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Difficulty levels
# ---------------------------------------------------------------------------


class DifficultyLevel(str, Enum):
    """Difficulty levels for generated questions."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    @property
    def label(self) -> str:
        return self.value.capitalize()

    def describe(self) -> str:
        descriptions = {
            DifficultyLevel.EASY: (
                "Suitable for beginners. Questions should cover fundamental concepts, "
                "simple syntax, basic data structures, and straightforward logic. "
                "Code snippets should be short (5-15 lines) and use common patterns."
            ),
            DifficultyLevel.MEDIUM: (
                "Suitable for intermediate developers. Questions should involve multiple "
                "concepts working together, moderate algorithmic thinking, design patterns, "
                "and debugging. Code snippets should be moderate length (15-40 lines) "
                "and may include edge cases."
            ),
            DifficultyLevel.HARD: (
                "Suitable for experienced developers. Questions should require deep "
                "understanding of complex systems, advanced algorithms, architectural "
                "trade-offs, performance considerations, and subtle bugs. Code snippets "
                "can be longer (40+ lines) and may involve concurrency, metaprogramming, "
                "or system design."
            ),
        }
        return descriptions[self]


# ---------------------------------------------------------------------------
# Topic categories
# ---------------------------------------------------------------------------


class Topic(str, Enum):
    """Topic categories for question generation."""

    ALGORITHMS = "algorithms"
    DATA_STRUCTURES = "data_structures"
    DESIGN_PATTERNS = "design_patterns"
    SYSTEM_DESIGN = "system_design"
    DEBUGGING = "debugging"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CONCURRENCY = "concurrency"
    CLEAN_CODE = "clean_code"
    API_DESIGN = "api_design"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()

    def describe(self) -> str:
        descriptions = {
            Topic.ALGORITHMS: (
                "Algorithmic problem-solving: sorting, searching, recursion, "
                "dynamic programming, graph traversal, and complexity analysis."
            ),
            Topic.DATA_STRUCTURES: (
                "Data structure fundamentals: arrays, linked lists, stacks, queues, "
                "trees, graphs, hash maps, and their time/space trade-offs."
            ),
            Topic.DESIGN_PATTERNS: (
                "Software design patterns: creational, structural, and behavioural "
                "patterns such as Singleton, Factory, Observer, Strategy, and their "
                "appropriate use cases."
            ),
            Topic.SYSTEM_DESIGN: (
                "System design and architecture: scalability, load balancing, caching, "
                "database design, microservices, and distributed systems concepts."
            ),
            Topic.DEBUGGING: (
                "Debugging and troubleshooting: identifying bugs, reading stack traces, "
                "understanding error messages, and systematic debugging strategies."
            ),
            Topic.SECURITY: (
                "Application security: input validation, authentication, authorisation, "
                "SQL injection, XSS, CSRF, and secure coding practices."
            ),
            Topic.PERFORMANCE: (
                "Performance optimisation: profiling, time complexity, memory usage, "
                "caching strategies, and efficient algorithm selection."
            ),
            Topic.CONCURRENCY: (
                "Concurrency and parallelism: threads, async/await, race conditions, "
                "deadlocks, mutexes, and concurrent data structures."
            ),
            Topic.CLEAN_CODE: (
                "Clean code and best practices: naming conventions, SOLID principles, "
                "DRY, code organisation, refactoring, and readability."
            ),
            Topic.API_DESIGN: (
                "API design principles: RESTful conventions, endpoint naming, "
                "versioning, error handling, pagination, and idempotency."
            ),
        }
        return descriptions[self]


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert code tutor. Your role is to generate thought-provoking \
questions that help developers deepen their understanding of code and \
software engineering concepts.

## Your Approach

1. **Present a code snippet or scenario** — Show real-world, practical code \
that illustrates the concept being tested. The code should be self-contained \
and runnable when possible.

2. **Ask a clear question** — Pose a specific question that requires the \
learner to analyse, trace, or reason about the code. Avoid yes/no questions; \
prefer questions that require explanation or prediction.

3. **Provide a detailed explanation** — After the question, provide a \
thorough explanation of the answer, including:
   - What the code does step by step
   - Why the correct answer is what it is
   - Common misconceptions or pitfalls
   - Practical takeaways or best practices

## Current Parameters

- **Difficulty**: {difficulty_label} — {difficulty_description}
- **Topic**: {topic_label} — {topic_description}
{language_line}{additional_context_block}
## Output Format

Respond with a JSON object matching this structure:

```json
{{
  "title": "A short, descriptive title for the question",
  "code_snippet": "The code snippet as a string",
  "language": "The programming language of the snippet",
  "question": "The question to ask the learner",
  "options": [
    {{"label": "A", "text": "First option"}},
    {{"label": "B", "text": "Second option"}},
    {{"label": "C", "text": "Third option"}},
    {{"label": "D", "text": "Fourth option"}}
  ],
  "correct_option": "A",
  "explanation": "Detailed explanation of the correct answer and why the others are wrong",
  "key_concepts": ["concept1", "concept2", "concept3"]
}}
```

## Guidelines

- The code snippet must be realistic and practical — not contrived or artificial.
- Multiple-choice options should be plausible; avoid obviously wrong answers.
- The explanation should teach, not just state the answer.
- Adapt complexity to the difficulty level precisely.
- If the topic is specific, ensure the question directly addresses it.
- Use idiomatic code for the chosen programming language.
- Include comments in the code only if they add educational value.
"""

ADDITIONAL_CONTEXT_BLOCK = """
## Additional Context

{additional_context}
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_question_prompt(
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
    topic: Topic = Topic.ALGORITHMS,
    language: str | None = None,
    additional_context: str | None = None,
) -> list[dict[str, str]]:
    """Build the message list for a question generation request.

    Args:
        difficulty: The difficulty level for the generated question.
        topic: The topic/category for the generated question.
        language: Optional programming language to target (e.g. "Python", "JavaScript").
        additional_context: Optional extra instructions or context for the LLM.

    Returns:
        A list of message dicts ready to pass to LLMClient.chat().
    """
    language_line = f"- **Language**: {language}" if language else ""
    additional_context_block = (
        ADDITIONAL_CONTEXT_BLOCK.format(additional_context=additional_context) if additional_context else ""
    )

    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        difficulty_label=difficulty.label,
        difficulty_description=difficulty.describe(),
        topic_label=topic.label,
        topic_description=topic.describe(),
        language_line=language_line,
        additional_context_block=additional_context_block,
    )

    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": (
                f"Generate a {difficulty.label.lower()}-difficulty question "
                f"about {topic.label}." + (f" The code should be in {language}." if language else "")
            ),
        },
    ]

    return messages


def build_explain_prompt(code: str, language: str | None = None) -> list[dict[str, str]]:
    """Build the message list for a code explanation request.

    Args:
        code: The code snippet to explain.
        language: Optional programming language hint.

    Returns:
        A list of message dicts ready to pass to LLMClient.chat().
    """
    language_hint = f" The code is written in {language}." if language else ""

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert code tutor. Your job is to explain code clearly "
                "and thoroughly. Break down what the code does step by step, explain "
                "key concepts, identify potential issues, and suggest improvements. "
                "Use plain language that a motivated learner can follow."
            ),
        },
        {
            "role": "user",
            "content": f"Please explain the following code:{language_hint}\n\n```\n{code}\n```",
        },
    ]

    return messages
