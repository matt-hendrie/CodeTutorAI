from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Form, Request
from fastapi.templating import Jinja2Templates
from openai import APITimeoutError, RateLimitError

from app.models.prompts import DifficultyLevel, QuestionFormat
from app.services.llm import llm_client
from app.services.prompts import DifficultyLevel as SvcDifficulty
from app.services.prompts import Topic, build_question_prompt

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/questions")
templates = Jinja2Templates(directory="app/templates")

# Mapping from model difficulty enums to service difficulty enums
DIFFICULTY_MAP = {
    DifficultyLevel.EASY: SvcDifficulty.EASY,
    DifficultyLevel.MEDIUM: SvcDifficulty.MEDIUM,
    DifficultyLevel.HARD: SvcDifficulty.HARD,
}

# Supported languages for the dropdown
LANGUAGES = [
    ("python", "Python"),
    ("javascript", "JavaScript"),
    ("typescript", "TypeScript"),
    ("java", "Java"),
    ("go", "Go"),
    ("rust", "Rust"),
    ("c", "C"),
    ("cpp", "C++"),
]


@router.get("/")
async def question_form(request: Request):
    """Render the question generation form page."""
    return templates.TemplateResponse(
        "pages/questions.html",
        {
            "request": request,
            "topics": list(Topic),
            "difficulties": list(DifficultyLevel),
            "formats": list(QuestionFormat),
            "languages": LANGUAGES,
            "selected_topic": Topic.ALGORITHMS,
            "selected_difficulty": DifficultyLevel.MEDIUM,
            "selected_format": QuestionFormat.SHORT_ANSWER,
            "selected_language": "python",
        },
    )


@router.post("/generate")
async def generate_question(
    request: Request,
    topic: Topic = Form(Topic.ALGORITHMS),
    difficulty: DifficultyLevel = Form(DifficultyLevel.MEDIUM),
    format: QuestionFormat = Form(QuestionFormat.SHORT_ANSWER),
    language: str = Form("python"),
):
    """Generate a question via HTMX and return an HTML partial."""
    # Map model difficulty to service difficulty (same values, different enums)
    svc_difficulty = DIFFICULTY_MAP.get(difficulty, SvcDifficulty.MEDIUM)

    # Build the prompt messages
    messages = build_question_prompt(
        difficulty=svc_difficulty,
        topic=topic,
        language=language,
    )

    # If the format is multiple choice, add instruction to the user message
    if format == QuestionFormat.MULTIPLE_CHOICE:
        messages[1]["content"] += " The question must be multiple choice with four options (A, B, C, D)."

    try:
        raw_response = await llm_client.chat(messages=messages, max_tokens=1024)
    except ValueError as exc:
        logger.warning("LLM client error: %s", exc)
        return templates.TemplateResponse(
            "partials/question_error.html",
            {"request": request, "error_message": str(exc)},
        )
    except RateLimitError:
        logger.warning("LLM rate limit exceeded")
        return templates.TemplateResponse(
            "partials/question_error.html",
            {
                "request": request,
                "error_message": "The AI model is currently rate-limited. Please wait a moment and try again.",
            },
        )
    except APITimeoutError:
        logger.warning("LLM request timed out")
        return templates.TemplateResponse(
            "partials/question_error.html",
            {
                "request": request,
                "error_message": "The request timed out. The AI model may be slow or unavailable — please try again.",
            },
        )
    except Exception as exc:
        logger.exception("Unexpected LLM error")
        return templates.TemplateResponse(
            "partials/question_error.html",
            {"request": request, "error_message": "An unexpected error occurred. Please try again."},
        )

    # Parse the LLM response — try to extract JSON from the response
    question_data = _parse_llm_response(raw_response)

    if question_data is None:
        logger.warning("Failed to parse LLM response as JSON")
        return templates.TemplateResponse(
            "partials/question_error.html",
            {
                "request": request,
                "error_message": "Could not parse the generated question. Please try again.",
                "raw_response": raw_response,
            },
        )

    return templates.TemplateResponse(
        "partials/question_result.html",
        {
            "request": request,
            "question": question_data,
            "difficulty": difficulty,
            "topic": topic,
            "format": format,
            "language": language,
        },
    )


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from around the text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove trailing ``` line if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def _extract_json_object(text: str) -> str | None:
    """Extract the outermost JSON object from text, handling garbage after the closing brace."""
    start = text.find("{")
    if start == -1:
        return None

    # Walk through the text counting braces to find the true end of the JSON object.
    # This handles cases where the LLM appends garbage after the closing brace.
    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\" and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # If we get here, the JSON is truncated (no closing brace found)
    return None


def _repair_truncated_json(text: str) -> str:
    """Attempt to repair truncated JSON by closing open strings, arrays, and objects.

    This handles the common case where the LLM runs out of tokens mid-response,
    leaving incomplete JSON. We try to close everything that's open.
    """
    # Track what's open so we can close it
    open_brackets = []  # stack of '{' or '['
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            escape_next = False
            continue

        if char == "\\" and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char in "{[":
            open_brackets.append(char)
        elif char in "}]":
            if open_brackets:
                # Check if it matches
                opener = open_brackets[-1]
                if (char == "}" and opener == "{") or (char == "]" and opener == "["):
                    open_brackets.pop()

    # Close any open string
    if in_string:
        text += '"'

    # Close any open brackets in reverse order
    for bracket in reversed(open_brackets):
        if bracket == "{":
            text += "}"
        elif bracket == "[":
            text += "]"

    return text


def _parse_llm_response(raw: str) -> dict | None:
    """Attempt to extract a JSON object from the LLM response.

    Handles:
    - Markdown code fences (```json ... ```)
    - Garbage text after the closing brace
    - Truncated JSON (missing closing braces/brackets)
    - JSON embedded in surrounding text

    Returns None if parsing fails.
    """
    text = _strip_markdown_fences(raw)

    # Strategy 1: Try parsing the whole text as JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract the JSON object (handles garbage after closing brace)
    extracted = _extract_json_object(text)
    if extracted:
        try:
            data = json.loads(extracted)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # Strategy 3: Try to repair truncated JSON
        repaired = _repair_truncated_json(extracted)
        if repaired != extracted:
            try:
                data = json.loads(repaired)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

    # Strategy 4: Extraction failed (truncated JSON with no closing brace).
    # Try to repair the full text directly.
    repaired = _repair_truncated_json(text)
    if repaired != text:
        try:
            data = json.loads(repaired)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None
