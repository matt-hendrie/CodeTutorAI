from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DifficultyLevel(str, Enum):
    """Difficulty levels for generated questions."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionTopic(str, Enum):
    """Topics/categories for question generation."""

    # Programming fundamentals
    VARIABLES = "variables"
    DATA_TYPES = "data_types"
    CONTROL_FLOW = "control_flow"
    FUNCTIONS = "functions"
    RECURSION = "recursion"
    ERROR_HANDLING = "error_handling"

    # Data structures
    ARRAYS = "arrays"
    LINKED_LISTS = "linked_lists"
    STACKS = "stacks"
    QUEUES = "queues"
    TREES = "trees"
    GRAPHS = "graphs"
    HASH_TABLES = "hash_tables"
    HEAPS = "heaps"

    # Object-oriented programming
    CLASSES = "classes"
    INHERITANCE = "inheritance"
    POLYMORPHISM = "polymorphism"
    DESIGN_PATTERNS = "design_patterns"

    # Architecture & design
    ALGORITHMS = "algorithms"
    SYSTEM_DESIGN = "system_design"
    API_DESIGN = "api_design"
    DATABASES = "databases"
    CONCURRENCY = "concurrency"

    # Web development
    HTTP = "http"
    REST = "rest"
    AUTHENTICATION = "authentication"

    # General
    DEBUGGING = "debugging"
    TESTING = "testing"
    PERFORMANCE = "performance"
    SECURITY = "security"


class QuestionFormat(str, Enum):
    """Output format for generated questions."""

    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_ANSWER = "short_answer"
    CODE_COMPLETION = "code_completion"
    CODE_REVIEW = "code_review"
    EXPLAIN_CODE = "explain_code"


class QuestionGenerationRequest(BaseModel):
    """Request model for generating a question."""

    topic: QuestionTopic = Field(
        ...,
        description="The topic/category for the question.",
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM,
        description="Difficulty level of the question.",
    )
    format: QuestionFormat = Field(
        default=QuestionFormat.SHORT_ANSWER,
        description="The format of the generated question.",
    )
    language: str = Field(
        default="python",
        description="Programming language context for the question.",
    )
    context: str = Field(
        default="",
        description="Optional additional context or code snippet to base the question on.",
    )
    count: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of questions to generate.",
    )


class ChoiceOption(BaseModel):
    """A single choice option for multiple-choice questions."""

    label: str = Field(..., description="The choice label, e.g. 'A', 'B', 'C', 'D'.")
    text: str = Field(..., description="The text of the choice.")
    is_correct: bool = Field(..., description="Whether this is the correct answer.")


class GeneratedQuestion(BaseModel):
    """A single generated question with its answer."""

    question: str = Field(..., description="The question text.")
    format: QuestionFormat = Field(..., description="The format of the question.")
    difficulty: DifficultyLevel = Field(..., description="The difficulty level.")
    topic: QuestionTopic = Field(..., description="The topic/category.")
    language: str = Field(default="python", description="The programming language context.")
    choices: list[ChoiceOption] = Field(
        default_factory=list,
        description="Choices for multiple-choice questions. Empty for other formats.",
    )
    answer: str = Field(..., description="The correct answer or explanation.")
    explanation: str = Field(
        default="",
        description="Detailed explanation of why the answer is correct.",
    )
    code_snippet: str = Field(
        default="",
        description="Code snippet associated with the question, if any.",
    )
    hints: list[str] = Field(
        default_factory=list,
        description="Optional hints to help the learner.",
    )


class QuestionGenerationResponse(BaseModel):
    """Response model for generated questions."""

    questions: list[GeneratedQuestion] = Field(
        ...,
        description="List of generated questions.",
    )
    model_used: str = Field(
        default="",
        description="The LLM model used to generate the questions.",
    )


# ---------------------------------------------------------------------------
# Answer evaluation models
# ---------------------------------------------------------------------------


class AnswerEvaluationRequest(BaseModel):
    """Request model for evaluating a user's answer to a question."""

    question: str = Field(
        ...,
        description="The original question text.",
    )
    correct_answer: str = Field(
        ...,
        description="The correct answer or explanation.",
    )
    user_answer: str = Field(
        ...,
        description="The user's submitted answer.",
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM,
        description="Difficulty level of the question.",
    )
    topic: QuestionTopic = Field(
        default=QuestionTopic.ALGORITHMS,
        description="The topic/category of the question.",
    )
    language: str = Field(
        default="python",
        description="Programming language context for the question.",
    )
    code_snippet: str = Field(
        default="",
        description="Code snippet associated with the question, if any.",
    )


class EvaluationScore(int):
    """Score range for answer evaluation (1-10 scale).

    1-3: Poor — fundamental misunderstanding or completely off-track
    4-5: Below average — partial understanding but significant gaps
    6-7: Satisfactory — mostly correct with minor gaps
    8-9: Good — strong understanding with small imperfections
    10: Excellent — complete and thorough understanding
    """

    MIN = 1
    MAX = 10


class AnswerEvaluationResponse(BaseModel):
    """Response model for an evaluated answer."""

    score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Score from 1 (poor) to 10 (excellent).",
    )
    score_label: str = Field(
        default="",
        description="Human-readable label for the score band (e.g. 'Satisfactory').",
    )
    is_correct: bool = Field(
        default=False,
        description="Whether the answer is essentially correct (score >= 7).",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="What the user got right in their answer.",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="What the user missed or got wrong.",
    )
    feedback: str = Field(
        default="",
        description="Constructive feedback explaining the evaluation.",
    )
    improved_answer: str = Field(
        default="",
        description="An improved version of the user's answer.",
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions for further learning.",
    )
    key_concepts: list[str] = Field(
        default_factory=list,
        description="Key concepts the user should review.",
    )
