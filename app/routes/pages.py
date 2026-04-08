from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse("pages/home.html", {"request": request})


@router.get("/demo")
async def demo(request: Request):
    """HTMX demo endpoint returning an HTML partial."""
    return templates.TemplateResponse(
        "partials/demo.html",
        {"request": request},
    )
