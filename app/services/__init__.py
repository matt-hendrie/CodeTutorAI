# Services package
from app.services.llm import LLMClient, llm_client
from app.services.prompts import (
    ADDITIONAL_CONTEXT_BLOCK,
    EVALUATE_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    DifficultyLevel,
    Topic,
    build_evaluate_prompt,
    build_explain_prompt,
    build_question_prompt,
)

__all__ = [
    "LLMClient",
    "llm_client",
    "ADDITIONAL_CONTEXT_BLOCK",
    "EVALUATE_PROMPT_TEMPLATE",
    "SYSTEM_PROMPT_TEMPLATE",
    "DifficultyLevel",
    "Topic",
    "build_evaluate_prompt",
    "build_explain_prompt",
    "build_question_prompt",
]
