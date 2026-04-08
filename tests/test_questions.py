"""Tests for question routes — form page, generate endpoint, error handling."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routes.questions import _parse_llm_response
from app.services.prompts import DifficultyLevel, Topic

client = TestClient(app)


# ---------------------------------------------------------------------------
# GET /questions/ — form page
# ---------------------------------------------------------------------------


class TestQuestionFormPage:
    """Tests for the question generation form page."""

    def test_form_page_returns_200(self):
        response = client.get("/questions/")
        assert response.status_code == 200

    def test_form_page_contains_topic_selector(self):
        response = client.get("/questions/")
        assert "Algorithms" in response.text
        assert "Data Structures" in response.text
        assert 'name="topic"' in response.text

    def test_form_page_contains_difficulty_buttons(self):
        response = client.get("/questions/")
        assert "Easy" in response.text
        assert "Medium" in response.text
        assert "Hard" in response.text
        assert 'name="difficulty"' in response.text

    def test_form_page_contains_format_selector(self):
        response = client.get("/questions/")
        assert 'name="format"' in response.text
        assert "Short Answer" in response.text
        assert "Multiple Choice" in response.text

    def test_form_page_contains_language_selector(self):
        response = client.get("/questions/")
        assert 'name="language"' in response.text
        assert "Python" in response.text
        assert "JavaScript" in response.text

    def test_form_page_contains_htmx_attributes(self):
        response = client.get("/questions/")
        assert 'hx-post="/questions/generate"' in response.text
        assert 'hx-target="#question-result"' in response.text
        assert 'hx-swap="innerHTML"' in response.text

    def test_form_page_contains_generate_button(self):
        response = client.get("/questions/")
        assert "Generate Question" in response.text

    def test_form_page_contains_all_10_topics(self):
        response = client.get("/questions/")
        for topic in Topic:
            assert topic.label in response.text

    def test_form_page_default_selections(self):
        response = client.get("/questions/")
        # Algorithms should be checked by default
        assert 'value="algorithms"' in response.text
        # Medium should be checked by default
        assert 'value="medium"' in response.text


# ---------------------------------------------------------------------------
# POST /questions/generate — generate endpoint
# ---------------------------------------------------------------------------


class TestGenerateQuestion:
    """Tests for the question generation endpoint."""

    @patch("app.routes.questions.llm_client")
    def test_generate_returns_question_on_success(self, mock_llm):
        mock_llm.chat = AsyncMock(
            return_value=json.dumps(
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
        )

        response = client.post(
            "/questions/generate",
            data={
                "topic": "algorithms",
                "difficulty": "medium",
                "format": "multiple_choice",
                "language": "python",
            },
        )
        assert response.status_code == 200
        assert "Binary Search" in response.text
        assert "Index of target" in response.text
        assert "medium" in response.text

    @patch("app.routes.questions.llm_client")
    def test_generate_with_easy_difficulty(self, mock_llm):
        mock_llm.chat = AsyncMock(
            return_value=json.dumps(
                {
                    "title": "Simple Variable",
                    "question": "What is x?",
                    "answer": "x is 5",
                    "explanation": "The variable x is assigned 5.",
                }
            )
        )

        response = client.post(
            "/questions/generate",
            data={
                "topic": "algorithms",
                "difficulty": "easy",
                "format": "short_answer",
                "language": "python",
            },
        )
        assert response.status_code == 200
        assert "Easy" in response.text

    @patch("app.routes.questions.llm_client")
    def test_generate_with_hard_difficulty(self, mock_llm):
        mock_llm.chat = AsyncMock(
            return_value=json.dumps(
                {
                    "title": "Concurrent Deadlock",
                    "question": "Why does this deadlock?",
                    "answer": "Lock ordering",
                    "explanation": "The locks are acquired in different orders.",
                }
            )
        )

        response = client.post(
            "/questions/generate",
            data={
                "topic": "concurrency",
                "difficulty": "hard",
                "format": "short_answer",
                "language": "java",
            },
        )
        assert response.status_code == 200
        assert "Hard" in response.text

    @patch("app.routes.questions.llm_client")
    def test_generate_with_markdown_wrapped_json(self, mock_llm):
        """LLM responses often wrap JSON in markdown code fences."""
        question_json = json.dumps(
            {
                "title": "Stack Operations",
                "question": "What is the output?",
                "answer": "3",
                "explanation": "The stack pops the last pushed element.",
            }
        )
        mock_llm.chat = AsyncMock(return_value=f"```json\n{question_json}\n```")

        response = client.post(
            "/questions/generate",
            data={
                "topic": "data_structures",
                "difficulty": "medium",
                "format": "short_answer",
                "language": "python",
            },
        )
        assert response.status_code == 200
        assert "Stack Operations" in response.text

    @patch("app.routes.questions.llm_client")
    def test_generate_with_surrounding_text(self, mock_llm):
        """LLM may include text before/after the JSON."""
        question_json = json.dumps(
            {
                "title": "Hash Map",
                "question": "What is the time complexity?",
                "answer": "O(1) average",
                "explanation": "Hash maps provide O(1) average lookup.",
            }
        )
        mock_llm.chat = AsyncMock(return_value=f"Here is your question:\n{question_json}\nHope that helps!")

        response = client.post(
            "/questions/generate",
            data={
                "topic": "data_structures",
                "difficulty": "easy",
                "format": "short_answer",
                "language": "python",
            },
        )
        assert response.status_code == 200
        assert "Hash Map" in response.text

    @patch("app.routes.questions.llm_client")
    def test_generate_error_on_missing_api_key(self, mock_llm):
        mock_llm.chat = AsyncMock(side_effect=ValueError("LLM API key is not configured"))

        response = client.post(
            "/questions/generate",
            data={
                "topic": "algorithms",
                "difficulty": "medium",
                "format": "short_answer",
                "language": "python",
            },
        )
        assert response.status_code == 200
        assert "Something went wrong" in response.text
        assert "LLM API key is not configured" in response.text

    @patch("app.routes.questions.llm_client")
    def test_generate_error_on_unexpected_exception(self, mock_llm):
        mock_llm.chat = AsyncMock(side_effect=RuntimeError("Connection refused"))

        response = client.post(
            "/questions/generate",
            data={
                "topic": "algorithms",
                "difficulty": "medium",
                "format": "short_answer",
                "language": "python",
            },
        )
        assert response.status_code == 200
        assert "Something went wrong" in response.text
        assert "unexpected error" in response.text

    @patch("app.routes.questions.llm_client")
    def test_generate_error_on_invalid_json(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value="This is not JSON at all, just plain text.")

        response = client.post(
            "/questions/generate",
            data={
                "topic": "algorithms",
                "difficulty": "medium",
                "format": "short_answer",
                "language": "python",
            },
        )
        assert response.status_code == 200
        assert "Something went wrong" in response.text
        assert "Could not parse" in response.text

    @patch("app.routes.questions.llm_client")
    def test_generate_shows_raw_response_on_parse_failure(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value="Not valid JSON")

        response = client.post(
            "/questions/generate",
            data={
                "topic": "algorithms",
                "difficulty": "medium",
                "format": "short_answer",
                "language": "python",
            },
        )
        assert response.status_code == 200
        assert "Not valid JSON" in response.text

    @patch("app.routes.questions.llm_client")
    def test_generate_passes_correct_parameters_to_llm(self, mock_llm):
        mock_llm.chat = AsyncMock(
            return_value=json.dumps(
                {
                    "title": "Test",
                    "question": "Test?",
                    "answer": "Test",
                    "explanation": "Test explanation",
                }
            )
        )

        client.post(
            "/questions/generate",
            data={
                "topic": "security",
                "difficulty": "hard",
                "format": "short_answer",
                "language": "rust",
            },
        )

        mock_llm.chat.assert_called_once()
        call_args = mock_llm.chat.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages")) or call_args[0][0]
        system_msg = messages[0]["content"]
        assert "Hard" in system_msg
        assert "Security" in system_msg
        user_msg = messages[1]["content"]
        assert "hard" in user_msg.lower()
        assert "Security" in user_msg
        assert "Rust" in user_msg or "rust" in user_msg

    @patch("app.routes.questions.llm_client")
    def test_generate_multiple_choice_adds_instruction(self, mock_llm):
        mock_llm.chat = AsyncMock(
            return_value=json.dumps(
                {
                    "title": "Test",
                    "question": "Test?",
                    "options": [
                        {"label": "A", "text": "A"},
                        {"label": "B", "text": "B"},
                        {"label": "C", "text": "C"},
                        {"label": "D", "text": "D"},
                    ],
                    "correct_option": "A",
                    "answer": "A",
                    "explanation": "Test",
                }
            )
        )

        client.post(
            "/questions/generate",
            data={
                "topic": "algorithms",
                "difficulty": "medium",
                "format": "multiple_choice",
                "language": "python",
            },
        )

        call_kwargs = mock_llm.chat.call_args.kwargs
        messages = call_kwargs.get("messages") or mock_llm.chat.call_args[1].get("messages")
        user_msg = messages[1]["content"]
        assert "multiple choice" in user_msg.lower()


# ---------------------------------------------------------------------------
# _parse_llm_response helper
# ---------------------------------------------------------------------------


class TestParseLLMResponse:
    """Tests for the _parse_llm_response helper function."""

    def test_parse_valid_json(self):
        data = {"title": "Test", "question": "What?"}
        result = _parse_llm_response(json.dumps(data))
        assert result == data

    def test_parse_json_in_markdown_code_block(self):
        data = {"title": "Test", "question": "What?"}
        raw = f"```json\n{json.dumps(data)}\n```"
        result = _parse_llm_response(raw)
        assert result == data

    def test_parse_json_in_plain_code_block(self):
        data = {"title": "Test", "question": "What?"}
        raw = f"```\n{json.dumps(data)}\n```"
        result = _parse_llm_response(raw)
        assert result == data

    def test_parse_json_with_surrounding_text(self):
        data = {"title": "Test", "question": "What?"}
        raw = f"Here is the question:\n{json.dumps(data)}\nEnjoy!"
        result = _parse_llm_response(raw)
        assert result == data

    def test_parse_returns_none_for_invalid_text(self):
        result = _parse_llm_response("This is just plain text with no JSON.")
        assert result is None

    def test_parse_returns_none_for_empty_string(self):
        result = _parse_llm_response("")
        assert result is None

    def test_parse_extracts_dict_from_json_array(self):
        """A JSON array containing a dict still yields the first dict found."""
        result = _parse_llm_response('[{"title": "Test"}]')
        assert result == {"title": "Test"}

    def test_parse_handles_nested_json(self):
        data = {
            "title": "Test",
            "options": [
                {"label": "A", "text": "Option A"},
                {"label": "B", "text": "Option B"},
            ],
        }
        result = _parse_llm_response(json.dumps(data))
        assert result == data
        assert len(result["options"]) == 2

    def test_parse_handles_whitespace(self):
        data = {"title": "Test"}
        result = _parse_llm_response(f"  \n  {json.dumps(data)}  \n  ")
        assert result == data
