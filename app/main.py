from collections import OrderedDict

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.auth import clear_auth_cookies, get_current_user, set_auth_cookies
from app.db import get_client

app = FastAPI(title="RoboScholar")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def _ctx(request: Request, **extra) -> dict:
    """Build template context with current user."""
    return {"request": request, "user": get_current_user(request), **extra}


# --- Pages ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    result = get_client().table("papers").select("*").order("number").execute()
    papers = result.data

    grouped: OrderedDict[str, list] = OrderedDict()
    for p in papers:
        grouped.setdefault(p["category"], []).append(p)

    return templates.TemplateResponse("home.html", _ctx(
        request, grouped_papers=grouped, total=len(papers),
    ))


@app.get("/papers/{slug}", response_class=HTMLResponse)
async def paper_detail(request: Request, slug: str):
    result = (
        get_client().table("papers").select("*")
        .eq("slug", slug).single().execute()
    )
    return templates.TemplateResponse("paper_detail.html", _ctx(
        request, paper=result.data,
    ))


# --- Auth ---

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", _ctx(request))


@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    try:
        client = get_client()
        resp = client.auth.sign_in_with_password({"email": email, "password": password})
        session = resp.session
        redirect = RedirectResponse("/", status_code=303)
        set_auth_cookies(redirect, session.model_dump())
        return redirect
    except Exception as e:
        return templates.TemplateResponse("login.html", _ctx(
            request, error=str(e),
        ))


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", _ctx(request))


@app.post("/signup")
async def signup(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    username: str = Form(...),
):
    try:
        client = get_client()
        resp = client.auth.sign_up({"email": email, "password": password})

        if resp.session:
            # Create profile
            from app.db import get_admin_client
            get_admin_client().table("profiles").insert({
                "id": resp.user.id,
                "username": username,
                "display_name": username,
            }).execute()

            redirect = RedirectResponse("/", status_code=303)
            set_auth_cookies(redirect, resp.session.model_dump())
            return redirect

        return templates.TemplateResponse("signup.html", _ctx(
            request, message="Check your email to confirm your account.",
        ))
    except Exception as e:
        return templates.TemplateResponse("signup.html", _ctx(
            request, error=str(e),
        ))


@app.get("/logout")
async def logout():
    redirect = RedirectResponse("/", status_code=303)
    clear_auth_cookies(redirect)
    return redirect
