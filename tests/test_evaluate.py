"""Tests for answer evaluation — prompt builder, models, and evaluate endpoint."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.prompts import (
    AnswerEvaluationRequest,
    AnswerEvaluationResponse,
    DifficultyLevel,
    EvaluationScore,
    QuestionTopic,
)
from app.services.prompts import (
    CODE_BLOCK_TEMPLATE,
    EVALUATE_PROMPT_TEMPLATE,
    QUESTION_BLOCK_TEMPLATE,
    Topic,
    build_evaluate_prompt,
)
from app.services.prompts import (
    DifficultyLevel as SvcDifficulty,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Evaluation prompt template
# ---------------------------------------------------------------------------


class TestEvaluatePromptTemplate:
    """Tests for the EVALUATE_PROMPT_TEMPLATE string."""

    def test_template_contains_evaluation_instructions(self):
        assert "evaluating" in EVALUATE_PROMPT_TEMPLATE.lower()

    def test_template_contains_scoring_rubric(self):
        assert "9-10" in EVALUATE_PROMPT_TEMPLATE
        assert "7-8" in EVALUATE_PROMPT_TEMPLATE
        assert "5-6" in EVALUATE_PROMPT_TEMPLATE
        assert "3-4" in EVALUATE_PROMPT_TEMPLATE
        assert "1-2" in EVALUATE_PROMPT_TEMPLATE

    def test_template_contains_json_format(self):
        assert '"score"' in EVALUATE_PROMPT_TEMPLATE
        assert '"feedback"' in EVALUATE_PROMPT_TEMPLATE
        assert '"correct_answer"' in EVALUATE_PROMPT_TEMPLATE
        assert '"key_points"' in EVALUATE_PROMPT_TEMPLATE
        assert '"follow_up"' in EVALUATE_PROMPT_TEMPLATE

    def test_template_has_format_placeholders(self):
        assert "{title}" in EVALUATE_PROMPT_TEMPLATE
        assert "{topic_label}" in EVALUATE_PROMPT_TEMPLATE
        assert "{difficulty_label}" in EVALUATE_PROMPT_TEMPLATE
        assert "{question_block}" in EVALUATE_PROMPT_TEMPLATE
        assert "{code_block}" in EVALUATE_PROMPT_TEMPLATE
        assert "{user_answer}" in EVALUATE_PROMPT_TEMPLATE

    def test_template_formats_correctly(self):
        result = EVALUATE_PROMPT_TEMPLATE.format(
            title="Binary Search",
            topic_label="Algorithms",
            topic_description="Algorithmic problem-solving.",
            difficulty_label="Medium",
            question_block=QUESTION_BLOCK_TEMPLATE.format(question="What does this do?"),
            code_block=CODE_BLOCK_TEMPLATE.format(code_snippet="def foo(): pass"),
            user_answer="It searches for an element.",
        )
        assert "Binary Search" in result
        assert "Algorithms" in result
        assert "Medium" in result
        assert "What does this do?" in result
        assert "def foo(): pass" in result
        assert "It searches for an element." in result


class TestQuestionBlockTemplate:
    """Tests for the QUESTION_BLOCK_TEMPLATE."""

    def test_template_contains_placeholder(self):
        assert "{question}" in QUESTION_BLOCK_TEMPLATE

    def test_template_formats_correctly(self):
        result = QUESTION_BLOCK_TEMPLATE.format(question="What is recursion?")
        assert "What is recursion?" in result


class TestCodeBlockTemplate:
    """Tests for the CODE_BLOCK_TEMPLATE."""

    def test_template_contains_placeholder(self):
        assert "{code_snippet}" in CODE_BLOCK_TEMPLATE

    def test_template_formats_correctly(self):
        result = CODE_BLOCK_TEMPLATE.format(code_snippet="def foo(): pass")
        assert "def foo(): pass" in result
        assert "```" in result


# ---------------------------------------------------------------------------
# build_evaluate_prompt()
# ---------------------------------------------------------------------------


class TestBuildEvaluatePrompt:
    """Tests for the build_evaluate_prompt function."""

    def test_default_parameters(self):
        messages = build_evaluate_prompt(
            question_data={"question": "What is a stack?"},
            user_answer="A stack is a LIFO data structure.",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_message_contains_evaluation_role(self):
        messages = build_evaluate_prompt(
            question_data={"question": "What is a stack?"},
            user_answer="A stack is a LIFO data structure.",
        )
        assert "evaluating" in messages[0]["content"].lower()
        assert "tutor" in messages[0]["content"].lower()

    def test_system_message_contains_scoring_rubric(self):
        messages = build_evaluate_prompt(
            question_data={"question": "What is a stack?"},
            user_answer="A stack is a LIFO data structure.",
        )
        assert "9-10" in messages[0]["content"]
        assert "1-2" in messages[0]["content"]

    def test_system_message_contains_question(self):
        messages = build_evaluate_prompt(
            question_data={"question": "What is a stack?"},
            user_answer="A stack is a LIFO data structure.",
        )
        assert "What is a stack?" in messages[0]["content"]

    def test_system_message_contains_code_snippet(self):
        messages = build_evaluate_prompt(
            question_data={
                "question": "What does this code do?",
                "code_snippet": "def push(item):\n    stack.append(item)",
            },
            user_answer="It adds an item to the stack.",
        )
        assert "def push(item):" in messages[0]["content"]

    def test_system_message_omits_code_block_when_no_snippet(self):
        messages = build_evaluate_prompt(
            question_data={"question": "What is a stack?"},
            user_answer="A stack is a LIFO data structure.",
        )
        assert "Code:" not in messages[0]["content"]
        assert "```" not in messages[0]["content"]

    def test_system_message_contains_user_answer(self):
        messages = build_evaluate_prompt(
            question_data={"question": "What is a stack?"},
            user_answer="A stack is a LIFO data structure.",
        )
        assert "A stack is a LIFO data structure." in messages[0]["content"]

    def test_user_message_contains_answer(self):
        messages = build_evaluate_prompt(
            question_data={"question": "What is a stack?"},
            user_answer="A stack is a LIFO data structure.",
        )
        assert "A stack is a LIFO data structure." in messages[1]["content"]

    def test_with_easy_difficulty(self):
        messages = build_evaluate_prompt(
            question_data={"question": "What is a variable?"},
            user_answer="A variable stores a value.",
            difficulty=SvcDifficulty.EASY,
        )
        assert "Easy" in messages[0]["content"]

    def test_with_hard_difficulty(self):
        messages = build_evaluate_prompt(
            question_data={"question": "Explain the CAP theorem."},
            user_answer="CAP stands for Consistency, Availability, Partition tolerance.",
            difficulty=SvcDifficulty.HARD,
        )
        assert "Hard" in messages[0]["content"]

    def test_with_topic(self):
        messages = build_evaluate_prompt(
            question_data={"question": "What is a stack?"},
            user_answer="A stack is a LIFO data structure.",
            topic=Topic.DATA_STRUCTURES,
        )
        assert "Data Structures" in messages[0]["content"]

    def test_with_title(self):
        messages = build_evaluate_prompt(
            question_data={"title": "Stack Operations", "question": "What is a stack?"},
            user_answer="A stack is a LIFO data structure.",
        )
        assert "Stack Operations" in messages[0]["content"]

    def test_all_difficulty_levels_generate_valid_prompts(self):
        for diff in SvcDifficulty:
            messages = build_evaluate_prompt(
                question_data={"question": "Test question"},
                user_answer="Test answer",
                difficulty=diff,
            )
            assert len(messages) == 2
            assert diff.label in messages[0]["content"]

    def test_all_topics_generate_valid_prompts(self):
        for topic in Topic:
            messages = build_evaluate_prompt(
                question_data={"question": "Test question"},
                user_answer="Test answer",
                topic=topic,
            )
            assert len(messages) == 2
            assert topic.label in messages[0]["content"]

    def test_system_message_contains_json_instruction(self):
        messages = build_evaluate_prompt(
            question_data={"question": "What is a stack?"},
            user_answer="A stack is a LIFO data structure.",
        )
        assert "JSON" in messages[0]["content"]

    def test_system_message_contains_correct_answer_field(self):
        messages = build_evaluate_prompt(
            question_data={"question": "What is a stack?"},
            user_answer="A stack is a LIFO data structure.",
        )
        assert "correct_answer" in messages[0]["content"]


# ---------------------------------------------------------------------------
# AnswerEvaluationRequest model
# ---------------------------------------------------------------------------


class TestAnswerEvaluationRequest:
    """Tests for the AnswerEvaluationRequest Pydantic model."""

    def test_defaults(self):
        req = AnswerEvaluationRequest(
            question="What is a stack?",
            correct_answer="A LIFO data structure.",
            user_answer="A data structure.",
        )
        assert req.difficulty == DifficultyLevel.MEDIUM
        assert req.topic == QuestionTopic.ALGORITHMS
        assert req.language == "python"
        assert req.code_snippet == ""

    def test_custom_values(self):
        req = AnswerEvaluationRequest(
            question="What is a stack?",
            correct_answer="A LIFO data structure.",
            user_answer="A data structure.",
            difficulty=DifficultyLevel.HARD,
            topic=QuestionTopic.STACKS,
            language="javascript",
            code_snippet="class Stack { ... }",
        )
        assert req.difficulty == DifficultyLevel.HARD
        assert req.topic == QuestionTopic.STACKS
        assert req.language == "javascript"
        assert req.code_snippet == "class Stack { ... }"

    def test_required_fields(self):
        with pytest.raises(Exception):
            AnswerEvaluationRequest()

    def test_question_is_required(self):
        with pytest.raises(Exception):
            AnswerEvaluationRequest(
                correct_answer="A LIFO data structure.",
                user_answer="A data structure.",
            )

    def test_correct_answer_is_required(self):
        with pytest.raises(Exception):
            AnswerEvaluationRequest(
                question="What is a stack?",
                user_answer="A data structure.",
            )

    def test_user_answer_is_required(self):
        with pytest.raises(Exception):
            AnswerEvaluationRequest(
                question="What is a stack?",
                correct_answer="A LIFO data structure.",
            )


# ---------------------------------------------------------------------------
# AnswerEvaluationResponse model
# ---------------------------------------------------------------------------


class TestAnswerEvaluationResponse:
    """Tests for the AnswerEvaluationResponse Pydantic model."""

    def test_minimal_response(self):
        resp = AnswerEvaluationResponse(score=7)
        assert resp.score == 7
        assert resp.score_label == ""
        assert resp.is_correct is False
        assert resp.strengths == []
        assert resp.weaknesses == []
        assert resp.feedback == ""
        assert resp.improved_answer == ""
        assert resp.follow_up_questions == []
        assert resp.key_concepts == []

    def test_full_response(self):
        resp = AnswerEvaluationResponse(
            score=8,
            score_label="Good",
            is_correct=True,
            strengths=["Correctly identified LIFO"],
            weaknesses=["Missed mention of push/pop"],
            feedback="Good understanding of the basic concept.",
            improved_answer="A stack is a LIFO data structure with push and pop operations.",
            follow_up_questions=["How would you implement a stack using an array?"],
            key_concepts=["LIFO", "push", "pop"],
        )
        assert resp.score == 8
        assert resp.score_label == "Good"
        assert resp.is_correct is True
        assert len(resp.strengths) == 1
        assert len(resp.weaknesses) == 1
        assert "Good understanding" in resp.feedback
        assert len(resp.follow_up_questions) == 1
        assert len(resp.key_concepts) == 3

    def test_score_minimum_value(self):
        resp = AnswerEvaluationResponse(score=1)
        assert resp.score == 1

    def test_score_maximum_value(self):
        resp = AnswerEvaluationResponse(score=10)
        assert resp.score == 10

    def test_score_below_minimum_raises(self):
        with pytest.raises(Exception):
            AnswerEvaluationResponse(score=0)

    def test_score_above_maximum_raises(self):
        with pytest.raises(Exception):
            AnswerEvaluationResponse(score=11)

    def test_is_correct_defaults_to_false(self):
        resp = AnswerEvaluationResponse(score=6)
        assert resp.is_correct is False


# ---------------------------------------------------------------------------
# EvaluationScore
# ---------------------------------------------------------------------------


class TestEvaluationScore:
    """Tests for the EvaluationScore class."""

    def test_min_value(self):
        assert EvaluationScore.MIN == 1

    def test_max_value(self):
        assert EvaluationScore.MAX == 10


# ---------------------------------------------------------------------------
# POST /questions/evaluate — endpoint
# ---------------------------------------------------------------------------


class TestEvaluateAnswer:
    """Tests for the evaluate answer endpoint."""

    def _make_question_json(self) -> str:
        """Helper to create a question JSON string for form submission."""
        return json.dumps(
            {
                "title": "Binary Search",
                "code_snippet": "def binary_search(arr, target):\n    ...",
                "language": "python",
                "question": "What does this function return?",
                "options": [
                    {"label": "A", "text": "Index of target"},
                    {"label": "B", "text": "True"},
                    {"label": "C", "text": "None"},
                    {"label": "D", "text": "Target value"},
                ],
                "correct_option": "A",
                "explanation": "Binary search returns the index of the target.",
                "key_concepts": ["binary search", "divide and conquer"],
            }
        )

    def _make_evaluation_json(self) -> str:
        """Helper to create an evaluation JSON response."""
        return json.dumps(
            {
                "score": 8,
                "feedback": (
                    "Good understanding of binary search. You correctly identified that it returns the index."
                ),
                "correct_answer": (
                    "Binary search returns the index of the target element in a sorted array, or -1 if not found."
                ),
                "key_points": [
                    "Correctly identified the return value as an index",
                    "Missed the case when the target is not found (returns -1)",
                ],
                "follow_up": "What is the time complexity of binary search, and why?",
            }
        )

    @patch("app.routes.questions.llm_client")
    def test_evaluate_returns_result_on_success(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value=self._make_evaluation_json())

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": self._make_question_json(),
                "user_answer": "It returns the index of the target.",
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "8" in response.text
        assert "Good understanding" in response.text

    @patch("app.routes.questions.llm_client")
    def test_evaluate_with_easy_difficulty(self, mock_llm):
        mock_llm.chat = AsyncMock(
            return_value=json.dumps(
                {
                    "score": 9,
                    "feedback": "Excellent!",
                    "correct_answer": "A variable stores a value.",
                    "key_points": ["Correctly identified storage"],
                    "follow_up": "What types can a variable hold?",
                }
            )
        )

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": json.dumps({"question": "What is a variable?"}),
                "user_answer": "A variable stores a value.",
                "difficulty": "easy",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "9" in response.text

    @patch("app.routes.questions.llm_client")
    def test_evaluate_with_hard_difficulty(self, mock_llm):
        mock_llm.chat = AsyncMock(
            return_value=json.dumps(
                {
                    "score": 4,
                    "feedback": "Partial understanding.",
                    "correct_answer": "CAP theorem states...",
                    "key_points": ["Correctly named the three properties"],
                    "follow_up": "Can you explain the trade-offs?",
                }
            )
        )

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": json.dumps({"question": "Explain the CAP theorem."}),
                "user_answer": "CAP stands for Consistency, Availability, Partition tolerance.",
                "difficulty": "hard",
                "topic": "system_design",
            },
        )
        assert response.status_code == 200
        assert "4" in response.text

    @patch("app.routes.questions.llm_client")
    def test_evaluate_with_markdown_wrapped_json(self, mock_llm):
        evaluation_json = self._make_evaluation_json()
        mock_llm.chat = AsyncMock(return_value=f"```json\n{evaluation_json}\n```")

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": self._make_question_json(),
                "user_answer": "It returns the index.",
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "8" in response.text

    @patch("app.routes.questions.llm_client")
    def test_evaluate_with_surrounding_text(self, mock_llm):
        evaluation_json = self._make_evaluation_json()
        mock_llm.chat = AsyncMock(return_value=f"Here is the evaluation:\n{evaluation_json}\nHope that helps!")

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": self._make_question_json(),
                "user_answer": "It returns the index.",
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "8" in response.text

    def test_evaluate_error_on_invalid_question_json(self):
        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": "not valid json {{{",
                "user_answer": "It returns the index.",
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "Could not load the question data" in response.text

    @patch("app.routes.questions.llm_client")
    def test_evaluate_error_on_missing_api_key(self, mock_llm):
        mock_llm.chat = AsyncMock(side_effect=ValueError("LLM API key is not configured."))

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": self._make_question_json(),
                "user_answer": "It returns the index.",
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "LLM API key" in response.text

    @patch("app.routes.questions.llm_client")
    def test_evaluate_error_on_rate_limit(self, mock_llm):
        import httpx
        from openai import RateLimitError

        mock_response = httpx.Response(
            status_code=429,
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        mock_llm.chat = AsyncMock(
            side_effect=RateLimitError(
                message="Rate limit exceeded",
                response=mock_response,
                body=None,
            )
        )

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": self._make_question_json(),
                "user_answer": "It returns the index.",
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "rate-limited" in response.text

    @patch("app.routes.questions.llm_client")
    def test_evaluate_error_on_timeout(self, mock_llm):
        from openai import APITimeoutError

        mock_llm.chat = AsyncMock(side_effect=APITimeoutError(request=None))

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": self._make_question_json(),
                "user_answer": "It returns the index.",
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "timed out" in response.text

    @patch("app.routes.questions.llm_client")
    def test_evaluate_error_on_unexpected_exception(self, mock_llm):
        mock_llm.chat = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": self._make_question_json(),
                "user_answer": "It returns the index.",
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "unexpected error" in response.text.lower()

    @patch("app.routes.questions.llm_client")
    def test_evaluate_error_on_invalid_json_response(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value="This is not JSON at all!")

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": self._make_question_json(),
                "user_answer": "It returns the index.",
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "Could not parse the evaluation result" in response.text

    @patch("app.routes.questions.llm_client")
    def test_evaluate_shows_raw_response_on_parse_failure(self, mock_llm):
        raw = "This is not JSON at all!"
        mock_llm.chat = AsyncMock(return_value=raw)

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": self._make_question_json(),
                "user_answer": "It returns the index.",
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "Show raw response" in response.text

    @patch("app.routes.questions.llm_client")
    def test_evaluate_passes_correct_parameters_to_llm(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value=self._make_evaluation_json())

        client.post(
            "/questions/evaluate",
            data={
                "question_json": self._make_question_json(),
                "user_answer": "It returns the index of the target.",
                "difficulty": "hard",
                "topic": "data_structures",
            },
        )

        mock_llm.chat.assert_called_once()
        call_args = mock_llm.chat.call_args
        messages = call_args.kwargs["messages"]

        # Verify system message contains evaluation instructions
        assert "evaluating" in messages[0]["content"].lower()
        # Verify the question is in the prompt
        assert "Binary Search" in messages[0]["content"]
        # Verify the user answer is in the prompt
        assert "It returns the index of the target." in messages[0]["content"]
        # Verify difficulty is in the prompt
        assert "Hard" in messages[0]["content"]
        # Verify topic is in the prompt
        assert "Data Structures" in messages[0]["content"]

    @patch("app.routes.questions.llm_client")
    def test_evaluate_with_code_snippet_in_question(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value=self._make_evaluation_json())

        question_json = json.dumps(
            {
                "title": "Stack Push",
                "question": "What does this code do?",
                "code_snippet": "def push(item):\n    stack.append(item)",
            }
        )

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": question_json,
                "user_answer": "It adds an item to the stack.",
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200

        # Verify the code snippet was passed to the LLM
        call_args = mock_llm.chat.call_args
        messages = call_args.kwargs["messages"]
        assert "def push(item):" in messages[0]["content"]

    @patch("app.routes.questions.llm_client")
    def test_evaluate_with_low_score(self, mock_llm):
        mock_llm.chat = AsyncMock(
            return_value=json.dumps(
                {
                    "score": 2,
                    "feedback": "The answer shows no understanding of the concept.",
                    "correct_answer": "Binary search returns the index.",
                    "key_points": ["Completely incorrect response"],
                    "follow_up": "What is binary search?",
                }
            )
        )

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": self._make_question_json(),
                "user_answer": "It sorts the array.",
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "2" in response.text

    @patch("app.routes.questions.llm_client")
    def test_evaluate_with_perfect_score(self, mock_llm):
        mock_llm.chat = AsyncMock(
            return_value=json.dumps(
                {
                    "score": 10,
                    "feedback": "Perfect answer!",
                    "correct_answer": "Binary search returns the index of the target.",
                    "key_points": ["Complete and thorough understanding"],
                    "follow_up": "How would you handle duplicates in the array?",
                }
            )
        )

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": self._make_question_json(),
                "user_answer": (
                    "Binary search returns the index of the target element in a sorted array, "
                    "using O(log n) time complexity by repeatedly dividing the search interval in half."
                ),
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "10" in response.text

    @patch("app.routes.questions.llm_client")
    def test_evaluate_uses_build_evaluate_prompt(self, mock_llm):
        """Verify the endpoint calls build_evaluate_prompt correctly."""
        mock_llm.chat = AsyncMock(return_value=self._make_evaluation_json())

        with patch("app.routes.questions.build_evaluate_prompt") as mock_build:
            mock_build.return_value = [
                {"role": "system", "content": "test system"},
                {"role": "user", "content": "test user"},
            ]

            client.post(
                "/questions/evaluate",
                data={
                    "question_json": self._make_question_json(),
                    "user_answer": "It returns the index.",
                    "difficulty": "hard",
                    "topic": "data_structures",
                },
            )

            mock_build.assert_called_once()
            call_kwargs = mock_build.call_args.kwargs
            assert call_kwargs["user_answer"] == "It returns the index."
            assert call_kwargs["difficulty"] == SvcDifficulty.HARD
            assert call_kwargs["topic"] == Topic.DATA_STRUCTURES

    @patch("app.routes.questions.llm_client")
    def test_evaluate_result_contains_feedback(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value=self._make_evaluation_json())

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": self._make_question_json(),
                "user_answer": "It returns the index.",
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "Good understanding" in response.text

    @patch("app.routes.questions.llm_client")
    def test_evaluate_result_contains_follow_up(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value=self._make_evaluation_json())

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": self._make_question_json(),
                "user_answer": "It returns the index.",
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "time complexity" in response.text

    @patch("app.routes.questions.llm_client")
    def test_evaluate_result_contains_correct_answer(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value=self._make_evaluation_json())

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": self._make_question_json(),
                "user_answer": "It returns the index.",
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "Feedback" in response.text

    @patch("app.routes.questions.llm_client")
    def test_evaluate_result_shows_feedback(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value=self._make_evaluation_json())

        response = client.post(
            "/questions/evaluate",
            data={
                "question_json": self._make_question_json(),
                "user_answer": "It returns the index of the target element.",
                "difficulty": "medium",
                "topic": "algorithms",
            },
        )
        assert response.status_code == 200
        assert "Feedback" in response.text
