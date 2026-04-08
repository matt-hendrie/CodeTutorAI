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
        raw_response = await llm_client.chat(messages=messages)
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


def _parse_llm_response(raw: str) -> dict | None:
    """Attempt to extract a JSON object from the LLM response.

    The LLM may wrap the JSON in markdown code blocks, so we strip those
    before parsing. Returns None if parsing fails.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find JSON object within the text as a fallback
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None
