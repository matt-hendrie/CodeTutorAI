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

    # LLM settings
    llm_api_key: str = ""
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Path settings
    templates_dir: str = "app/templates"
    static_dir: str = "app/static"


settings = Settings()
