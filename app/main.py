from collections import OrderedDict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.db import get_client

app = FastAPI(title="RoboScholar")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    result = get_client().table("papers").select("*").order("number").execute()
    papers = result.data

    # Group by category, preserving order
    grouped: OrderedDict[str, list] = OrderedDict()
    for p in papers:
        cat = p["category"]
        grouped.setdefault(cat, []).append(p)

    return templates.TemplateResponse("home.html", {
        "request": request,
        "grouped_papers": grouped,
        "total": len(papers),
    })


@app.get("/papers/{slug}", response_class=HTMLResponse)
async def paper_detail(request: Request, slug: str):
    result = (
        get_client()
        .table("papers")
        .select("*")
        .eq("slug", slug)
        .single()
        .execute()
    )
    paper = result.data
    return templates.TemplateResponse("paper_detail.html", {
        "request": request,
        "paper": paper,
    })
