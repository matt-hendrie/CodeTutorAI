# Services package
from app.services.llm import LLMClient, llm_client
from app.services.prompts import (
    ADDITIONAL_CONTEXT_BLOCK,
    SYSTEM_PROMPT_TEMPLATE,
    DifficultyLevel,
    Topic,
    build_explain_prompt,
    build_question_prompt,
)

__all__ = [
    "LLMClient",
    "llm_client",
    "ADDITIONAL_CONTEXT_BLOCK",
    "SYSTEM_PROMPT_TEMPLATE",
    "DifficultyLevel",
    "Topic",
    "build_explain_prompt",
    "build_question_prompt",
]
