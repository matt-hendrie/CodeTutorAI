from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_prefix="CODETUTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Application settings
    app_name: str = "CodeTutorAI"
    app_version: str = "0.1.0"
    debug: bool = False

    # LLM settings (OpenRouter)
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    llm_site_url: str = ""
    llm_site_name: str = "CodeTutorAI"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Frontend CDN settings
    htmx_version: str = "2.0.4"
    tailwind_enabled: bool = True

    # Path settings
    templates_dir: str = "app/templates"
    static_dir: str = "app/static"


settings = Settings()
