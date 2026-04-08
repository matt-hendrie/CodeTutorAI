from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.routes import pages


def create_app() -> FastAPI:
    """Application factory for the CodeTutorAI FastAPI app."""

    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
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
