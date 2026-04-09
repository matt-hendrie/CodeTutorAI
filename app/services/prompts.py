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
You are a code tutor. Generate a question about code and software engineering.

Difficulty: {difficulty_label} — {difficulty_description}
Topic: {topic_label} — {topic_description}
{language_line}{additional_context_block}
Respond ONLY with valid JSON — no markdown, no extra text. Format:

{{"title": "short title", "code_snippet": "the code", "language": "python", "question": "the question", "options": [{{"label": "A", "text": "option"}}, {{"label": "B", "text": "option"}}, {{"label": "C", "text": "option"}}, {{"label": "D", "text": "option"}}], "correct_option": "A", "explanation": "why A is correct", "key_concepts": ["concept1", "concept2"]}}

Rules:
- Code must be realistic and runnable
- Options must be plausible, not obviously wrong
- Explanation should teach, not just state the answer
- Keep the response concise — do not exceed the JSON structure above
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


# ---------------------------------------------------------------------------
# Answer evaluation prompt
# ---------------------------------------------------------------------------

EVALUATE_PROMPT_TEMPLATE = """\
You are a code tutor evaluating a learner's answer. Be fair, constructive, and encouraging.

## Question
Title: {title}
Topic: {topic_label} — {topic_description}
Difficulty: {difficulty_label}

{question_block}{code_block}
## Learner's Answer
{user_answer}

## Your Task
Evaluate the answer and respond ONLY with valid JSON — no markdown, no extra text. Format:

{{"score": 7, "feedback": "Specific, constructive feedback about what was right and what was wrong. Be encouraging but honest.", "correct_answer": "A concise summary of the correct answer", "key_points": ["what the learner got right", "what they missed or got wrong"], "follow_up": "A follow-up question to deepen understanding"}}

## Scoring Rubric
- 9-10: Excellent — fully correct with nuance and depth
- 7-8: Good — mostly correct with minor gaps
- 5-6: Partial — some understanding but significant gaps
- 3-4: Weak — limited understanding, mostly incorrect
- 1-2: Incorrect — no meaningful understanding demonstrated

## Rules
- Score must be an integer from 1 to 10
- Feedback must be specific — reference what the learner said
- If the answer is completely wrong, still be encouraging and explain the correct approach
- The follow_up question should help the learner explore the topic further
- Keep the response concise
"""

QUESTION_BLOCK_TEMPLATE = """
Question: {question}
"""

CODE_BLOCK_TEMPLATE = """
Code:
```
{code_snippet}
```
"""


def build_evaluate_prompt(
    question_data: dict[str, Any],
    user_answer: str,
    difficulty: DifficultyLevel | None = None,
    topic: Topic | None = None,
) -> list[dict[str, str]]:
    """Build the message list for evaluating a learner's answer.

    Args:
        question_data: The question dict (from LLM response) containing title,
            question, code_snippet, correct_option, explanation, etc.
        user_answer: The learner's submitted answer text.
        difficulty: Optional difficulty level for context.
        topic: Optional topic for context.

    Returns:
        A list of message dicts ready to pass to LLMClient.chat().
    """
    # Determine difficulty and topic from question_data or parameters
    diff = difficulty or DifficultyLevel.MEDIUM
    top = topic or Topic.ALGORITHMS

    question_block = QUESTION_BLOCK_TEMPLATE.format(question=question_data.get("question", ""))
    code_snippet = question_data.get("code_snippet", "")
    code_block = CODE_BLOCK_TEMPLATE.format(code_snippet=code_snippet) if code_snippet else ""

    system_content = EVALUATE_PROMPT_TEMPLATE.format(
        title=question_data.get("title", ""),
        topic_label=top.label,
        topic_description=top.describe(),
        difficulty_label=diff.label,
        question_block=question_block,
        code_block=code_block,
        user_answer=user_answer,
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Evaluate this answer:\n\n{user_answer}"},
    ]

    return messages
