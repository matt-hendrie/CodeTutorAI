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
