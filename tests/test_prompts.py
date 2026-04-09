"""Tests for prompt models, difficulty levels, topics, and prompt builder."""

import pytest

from app.models.prompts import (
    ChoiceOption,
    DifficultyLevel,
    GeneratedQuestion,
    QuestionFormat,
    QuestionGenerationRequest,
    QuestionGenerationResponse,
    QuestionTopic,
)
from app.services.prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    Topic,
    build_explain_prompt,
    build_question_prompt,
)
from app.services.prompts import (
    DifficultyLevel as SvcDifficulty,
)

# ---------------------------------------------------------------------------
# DifficultyLevel (models)
# ---------------------------------------------------------------------------


class TestDifficultyLevel:
    """Tests for the DifficultyLevel enum in models."""

    def test_values(self):
        assert DifficultyLevel.EASY == "easy"
        assert DifficultyLevel.MEDIUM == "medium"
        assert DifficultyLevel.HARD == "hard"

    def test_member_count(self):
        assert len(DifficultyLevel) == 3

    def test_from_string(self):
        assert DifficultyLevel("easy") == DifficultyLevel.EASY
        assert DifficultyLevel("medium") == DifficultyLevel.MEDIUM
        assert DifficultyLevel("hard") == DifficultyLevel.HARD

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            DifficultyLevel("extreme")


# ---------------------------------------------------------------------------
# QuestionTopic (models)
# ---------------------------------------------------------------------------


class TestQuestionTopic:
    """Tests for the QuestionTopic enum in models."""

    def test_fundamental_topics(self):
        assert QuestionTopic.VARIABLES == "variables"
        assert QuestionTopic.DATA_TYPES == "data_types"
        assert QuestionTopic.CONTROL_FLOW == "control_flow"
        assert QuestionTopic.FUNCTIONS == "functions"
        assert QuestionTopic.RECURSION == "recursion"
        assert QuestionTopic.ERROR_HANDLING == "error_handling"

    def test_data_structure_topics(self):
        assert QuestionTopic.ARRAYS == "arrays"
        assert QuestionTopic.LINKED_LISTS == "linked_lists"
        assert QuestionTopic.STACKS == "stacks"
        assert QuestionTopic.QUEUES == "queues"
        assert QuestionTopic.TREES == "trees"
        assert QuestionTopic.GRAPHS == "graphs"
        assert QuestionTopic.HASH_TABLES == "hash_tables"
        assert QuestionTopic.HEAPS == "heaps"

    def test_oop_topics(self):
        assert QuestionTopic.CLASSES == "classes"
        assert QuestionTopic.INHERITANCE == "inheritance"
        assert QuestionTopic.POLYMORPHISM == "polymorphism"
        assert QuestionTopic.DESIGN_PATTERNS == "design_patterns"

    def test_architecture_topics(self):
        assert QuestionTopic.ALGORITHMS == "algorithms"
        assert QuestionTopic.SYSTEM_DESIGN == "system_design"
        assert QuestionTopic.API_DESIGN == "api_design"
        assert QuestionTopic.DATABASES == "databases"
        assert QuestionTopic.CONCURRENCY == "concurrency"

    def test_web_topics(self):
        assert QuestionTopic.HTTP == "http"
        assert QuestionTopic.REST == "rest"
        assert QuestionTopic.AUTHENTICATION == "authentication"

    def test_general_topics(self):
        assert QuestionTopic.DEBUGGING == "debugging"
        assert QuestionTopic.TESTING == "testing"
        assert QuestionTopic.PERFORMANCE == "performance"
        assert QuestionTopic.SECURITY == "security"

    def test_total_topic_count(self):
        assert len(QuestionTopic) == 30


# ---------------------------------------------------------------------------
# QuestionFormat (models)
# ---------------------------------------------------------------------------


class TestQuestionFormat:
    """Tests for the QuestionFormat enum in models."""

    def test_values(self):
        assert QuestionFormat.MULTIPLE_CHOICE == "multiple_choice"
        assert QuestionFormat.SHORT_ANSWER == "short_answer"
        assert QuestionFormat.CODE_COMPLETION == "code_completion"
        assert QuestionFormat.CODE_REVIEW == "code_review"
        assert QuestionFormat.EXPLAIN_CODE == "explain_code"

    def test_member_count(self):
        assert len(QuestionFormat) == 5


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestQuestionGenerationRequest:
    """Tests for the QuestionGenerationRequest model."""

    def test_defaults(self):
        req = QuestionGenerationRequest(topic=QuestionTopic.ALGORITHMS)
        assert req.topic == QuestionTopic.ALGORITHMS
        assert req.difficulty == DifficultyLevel.MEDIUM
        assert req.format == QuestionFormat.SHORT_ANSWER
        assert req.language == "python"
        assert req.context == ""
        assert req.count == 1

    def test_custom_values(self):
        req = QuestionGenerationRequest(
            topic=QuestionTopic.DESIGN_PATTERNS,
            difficulty=DifficultyLevel.HARD,
            format=QuestionFormat.MULTIPLE_CHOICE,
            language="javascript",
            context="Focus on the Observer pattern",
            count=5,
        )
        assert req.topic == QuestionTopic.DESIGN_PATTERNS
        assert req.difficulty == DifficultyLevel.HARD
        assert req.format == QuestionFormat.MULTIPLE_CHOICE
        assert req.language == "javascript"
        assert req.context == "Focus on the Observer pattern"
        assert req.count == 5

    def test_count_validation_ge_1(self):
        with pytest.raises(ValueError):
            QuestionGenerationRequest(topic=QuestionTopic.ALGORITHMS, count=0)

    def test_count_validation_le_10(self):
        with pytest.raises(ValueError):
            QuestionGenerationRequest(topic=QuestionTopic.ALGORITHMS, count=11)

    def test_count_boundary_values(self):
        req_min = QuestionGenerationRequest(topic=QuestionTopic.ALGORITHMS, count=1)
        assert req_min.count == 1
        req_max = QuestionGenerationRequest(topic=QuestionTopic.ALGORITHMS, count=10)
        assert req_max.count == 10


class TestChoiceOption:
    """Tests for the ChoiceOption model."""

    def test_basic_choice(self):
        choice = ChoiceOption(label="A", text="Option A", is_correct=True)
        assert choice.label == "A"
        assert choice.text == "Option A"
        assert choice.is_correct is True

    def test_incorrect_choice(self):
        choice = ChoiceOption(label="B", text="Option B", is_correct=False)
        assert choice.is_correct is False


class TestGeneratedQuestion:
    """Tests for the GeneratedQuestion model."""

    def test_minimal_question(self):
        q = GeneratedQuestion(
            question="What does this code do?",
            format=QuestionFormat.SHORT_ANSWER,
            difficulty=DifficultyLevel.EASY,
            topic=QuestionTopic.FUNCTIONS,
            answer="It defines a function.",
        )
        assert q.question == "What does this code do?"
        assert q.format == QuestionFormat.SHORT_ANSWER
        assert q.difficulty == DifficultyLevel.EASY
        assert q.topic == QuestionTopic.FUNCTIONS
        assert q.language == "python"
        assert q.choices == []
        assert q.answer == "It defines a function."
        assert q.explanation == ""
        assert q.code_snippet == ""
        assert q.hints == []

    def test_full_question(self):
        q = GeneratedQuestion(
            question="Which design pattern is this?",
            format=QuestionFormat.MULTIPLE_CHOICE,
            difficulty=DifficultyLevel.HARD,
            topic=QuestionTopic.DESIGN_PATTERNS,
            language="java",
            choices=[
                ChoiceOption(label="A", text="Singleton", is_correct=False),
                ChoiceOption(label="B", text="Observer", is_correct=True),
                ChoiceOption(label="C", text="Factory", is_correct=False),
                ChoiceOption(label="D", text="Strategy", is_correct=False),
            ],
            answer="B",
            explanation="The Observer pattern defines a one-to-many dependency...",
            code_snippet="public class EventManager { ... }",
            hints=["Think about publish-subscribe", "Look at the notify method"],
        )
        assert len(q.choices) == 4
        assert q.choices[1].is_correct is True
        assert q.language == "java"
        assert len(q.hints) == 2


class TestQuestionGenerationResponse:
    """Tests for the QuestionGenerationResponse model."""

    def test_response_with_questions(self):
        q = GeneratedQuestion(
            question="Test?",
            format=QuestionFormat.SHORT_ANSWER,
            difficulty=DifficultyLevel.MEDIUM,
            topic=QuestionTopic.ALGORITHMS,
            answer="Test answer",
        )
        resp = QuestionGenerationResponse(questions=[q], model_used="gpt-4")
        assert len(resp.questions) == 1
        assert resp.model_used == "gpt-4"

    def test_default_model_used(self):
        resp = QuestionGenerationResponse(
            questions=[
                GeneratedQuestion(
                    question="Q",
                    format=QuestionFormat.SHORT_ANSWER,
                    difficulty=DifficultyLevel.EASY,
                    topic=QuestionTopic.VARIABLES,
                    answer="A",
                )
            ]
        )
        assert resp.model_used == ""


# ---------------------------------------------------------------------------
# DifficultyLevel (services) — with label and describe
# ---------------------------------------------------------------------------


class TestServiceDifficultyLevel:
    """Tests for the DifficultyLevel enum in services (with label/describe)."""

    def test_values(self):
        from app.services.prompts import DifficultyLevel as SvcDifficulty

        assert SvcDifficulty.EASY == "easy"
        assert SvcDifficulty.MEDIUM == "medium"
        assert SvcDifficulty.HARD == "hard"

    def test_label(self):
        from app.services.prompts import DifficultyLevel as SvcDifficulty

        assert SvcDifficulty.EASY.label == "Easy"
        assert SvcDifficulty.MEDIUM.label == "Medium"
        assert SvcDifficulty.HARD.label == "Hard"

    def test_describe_returns_non_empty(self):
        from app.services.prompts import DifficultyLevel as SvcDifficulty

        for level in SvcDifficulty:
            assert len(level.describe()) > 20, f"{level} description too short"


# ---------------------------------------------------------------------------
# Topic (services) — with label and describe
# ---------------------------------------------------------------------------


class TestServiceTopic:
    """Tests for the Topic enum in services (with label/describe)."""

    def test_topic_count(self):
        assert len(Topic) == 10

    def test_all_topics_have_descriptions(self):
        for topic in Topic:
            assert len(topic.describe()) > 20, f"{topic} description too short"

    def test_label_formatting(self):
        assert Topic.ALGORITHMS.label == "Algorithms"
        assert Topic.DATA_STRUCTURES.label == "Data Structures"
        assert Topic.DESIGN_PATTERNS.label == "Design Patterns"
        assert Topic.CLEAN_CODE.label == "Clean Code"
        assert Topic.API_DESIGN.label == "Api Design"

    def test_all_topics(self):
        expected = {
            "algorithms",
            "data_structures",
            "design_patterns",
            "system_design",
            "debugging",
            "security",
            "performance",
            "concurrency",
            "clean_code",
            "api_design",
        }
        actual = {t.value for t in Topic}
        assert actual == expected


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT_TEMPLATE
# ---------------------------------------------------------------------------


class TestSystemPromptTemplate:
    """Tests for the SYSTEM_PROMPT_TEMPLATE string."""

    def test_template_contains_key_sections(self):
        assert "code tutor" in SYSTEM_PROMPT_TEMPLATE.lower()
        assert "Difficulty" in SYSTEM_PROMPT_TEMPLATE
        assert "Topic" in SYSTEM_PROMPT_TEMPLATE
        assert "JSON" in SYSTEM_PROMPT_TEMPLATE
        assert "Rules" in SYSTEM_PROMPT_TEMPLATE

    def test_template_has_format_placeholders(self):
        assert "{difficulty_label}" in SYSTEM_PROMPT_TEMPLATE
        assert "{difficulty_description}" in SYSTEM_PROMPT_TEMPLATE
        assert "{topic_label}" in SYSTEM_PROMPT_TEMPLATE
        assert "{topic_description}" in SYSTEM_PROMPT_TEMPLATE
        assert "{language_line}" in SYSTEM_PROMPT_TEMPLATE
        assert "{additional_context_block}" in SYSTEM_PROMPT_TEMPLATE

    def test_template_formats_correctly(self):
        from app.services.prompts import DifficultyLevel as SvcDifficulty

        result = SYSTEM_PROMPT_TEMPLATE.format(
            difficulty_label=SvcDifficulty.EASY.label,
            difficulty_description=SvcDifficulty.EASY.describe(),
            topic_label=Topic.ALGORITHMS.label,
            topic_description=Topic.ALGORITHMS.describe(),
            language_line="",
            additional_context_block="",
        )
        assert "Easy" in result
        assert "Algorithms" in result
        assert "code tutor" in result.lower()


# ---------------------------------------------------------------------------
# build_question_prompt
# ---------------------------------------------------------------------------


class TestBuildQuestionPrompt:
    """Tests for the build_question_prompt function."""

    def test_default_parameters(self):
        messages = build_question_prompt()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "medium" in messages[1]["content"]
        assert "Algorithms" in messages[1]["content"]

    def test_easy_difficulty(self):
        messages = build_question_prompt(difficulty=SvcDifficulty.EASY)
        assert "Easy" in messages[0]["content"]
        assert "easy-difficulty" in messages[1]["content"]

    def test_hard_difficulty(self):
        messages = build_question_prompt(difficulty=SvcDifficulty.HARD)
        assert "Hard" in messages[0]["content"]
        assert "hard-difficulty" in messages[1]["content"]

    def test_with_topic(self):
        messages = build_question_prompt(topic=Topic.SECURITY)
        assert "Security" in messages[0]["content"]
        assert "Security" in messages[1]["content"]

    def test_with_language(self):
        messages = build_question_prompt(language="JavaScript")
        assert "**Language**: JavaScript" in messages[0]["content"]
        assert "JavaScript" in messages[1]["content"]

    def test_without_language(self):
        messages = build_question_prompt(language=None)
        assert "**Language**" not in messages[0]["content"]

    def test_with_additional_context(self):
        messages = build_question_prompt(additional_context="Focus on async/await patterns")
        assert "Additional Context" in messages[0]["content"]
        assert "async/await patterns" in messages[0]["content"]

    def test_without_additional_context(self):
        messages = build_question_prompt(additional_context=None)
        assert "Additional Context" not in messages[0]["content"]

    def test_system_message_contains_role_definition(self):
        messages = build_question_prompt()
        assert "code tutor" in messages[0]["content"].lower()

    def test_system_message_contains_output_format(self):
        messages = build_question_prompt()
        assert "JSON" in messages[0]["content"]
        assert "title" in messages[0]["content"]
        assert "code_snippet" in messages[0]["content"]
        assert "correct_option" in messages[0]["content"]
        assert "explanation" in messages[0]["content"]

    def test_all_difficulty_levels_generate_valid_prompts(self):
        for level in SvcDifficulty:
            messages = build_question_prompt(difficulty=level)
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"

    def test_all_topics_generate_valid_prompts(self):
        for topic in Topic:
            messages = build_question_prompt(topic=topic)
            assert len(messages) == 2
            assert topic.label in messages[0]["content"]


# ---------------------------------------------------------------------------
# build_explain_prompt
# ---------------------------------------------------------------------------


class TestBuildExplainPrompt:
    """Tests for the build_explain_prompt function."""

    def test_basic_prompt(self):
        code = "def add(a, b):\n    return a + b"
        messages = build_explain_prompt(code=code)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert code in messages[1]["content"]

    def test_with_language(self):
        code = "const x = 1;"
        messages = build_explain_prompt(code=code, language="JavaScript")
        assert "JavaScript" in messages[1]["content"]

    def test_without_language(self):
        code = "x = 1"
        messages = build_explain_prompt(code=code, language=None)
        assert "written in" not in messages[1]["content"]

    def test_system_message_is_tutor(self):
        code = "print('hello')"
        messages = build_explain_prompt(code=code)
        assert "code tutor" in messages[0]["content"].lower()
        assert "explain" in messages[0]["content"].lower()

    def test_code_in_code_block(self):
        code = "x = 42"
        messages = build_explain_prompt(code=code)
        assert f"```\n{code}\n```" in messages[1]["content"]

    def test_multiline_code(self):
        code = "def greet(name):\n    return f'Hello, {name}!'"
        messages = build_explain_prompt(code=code)
        assert code in messages[1]["content"]
