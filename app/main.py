from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.routes import pages
from app.services.llm import llm_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan — startup and shutdown."""
    yield
    # Shutdown: close the LLM client connection
    await llm_client.close()


def create_app() -> FastAPI:
    """Application factory for the CodeTutorAI FastAPI app."""

    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Mount static files
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    # Configure Jinja2 templates
    templates = Jinja2Templates(directory="app/templates")
    app.state.templates = templates

    # Include routers
    app.include_router(pages.router)

    return app


app = create_app()
