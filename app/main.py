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


# --- Quiz ---

@app.get("/papers/{slug}/quiz", response_class=HTMLResponse)
async def quiz_page(request: Request, slug: str):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    db = get_client()
    paper = db.table("papers").select("*").eq("slug", slug).single().execute().data
    questions = (
        db.table("questions").select("*")
        .eq("paper_number", paper["number"]).execute().data
    )

    if not questions:
        return templates.TemplateResponse("quiz.html", _ctx(
            request, paper=paper, question=None, total=0, current=0,
        ))

    # Find first unanswered question for this user
    attempts = (
        db.table("quiz_attempts").select("question_id")
        .eq("user_id", user["id"]).execute().data
    )
    answered_ids = set(a["question_id"] for a in attempts)
    unanswered = [q for q in questions if q["id"] not in answered_ids]

    import json
    question = unanswered[0] if unanswered else None
    if question and isinstance(question["choices"], str):
        question["choices"] = json.loads(question["choices"])

    return templates.TemplateResponse("quiz.html", _ctx(
        request,
        paper=paper,
        question=question,
        total=len(questions),
        answered=len(answered_ids),
        done=not unanswered,
    ))


@app.post("/papers/{slug}/quiz")
async def quiz_submit(
    request: Request,
    slug: str,
    question_id: int = Form(...),
    selected: int = Form(...),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    db = get_client()
    paper = db.table("papers").select("*").eq("slug", slug).single().execute().data

    import json
    question = db.table("questions").select("*").eq("id", question_id).single().execute().data
    if isinstance(question["choices"], str):
        question["choices"] = json.loads(question["choices"])

    is_correct = selected == question["correct_index"]

    # Record attempt
    from app.db import get_admin_client
    admin = get_admin_client()
    admin.table("quiz_attempts").insert({
        "user_id": user["id"],
        "question_id": question_id,
        "selected_index": selected,
        "is_correct": is_correct,
    }).execute()

    # Update paper progress
    points_earned = 10 if is_correct else 0
    progress = (
        admin.table("user_paper_progress")
        .select("*")
        .eq("user_id", user["id"])
        .eq("paper_number", paper["number"])
        .execute().data
    )

    total_questions = len(
        db.table("questions").select("id")
        .eq("paper_number", paper["number"]).execute().data
    )

    if progress:
        p = progress[0]
        new_answered = p["questions_answered"] + 1
        new_correct = p["questions_correct"] + (1 if is_correct else 0)
        new_points = p["points"] + points_earned
        completed = new_answered >= total_questions
        admin.table("user_paper_progress").update({
            "questions_answered": new_answered,
            "questions_correct": new_correct,
            "points": new_points,
            "completed": completed,
            "updated_at": "now()",
        }).eq("id", p["id"]).execute()
    else:
        admin.table("user_paper_progress").insert({
            "user_id": user["id"],
            "paper_number": paper["number"],
            "questions_answered": 1,
            "questions_correct": 1 if is_correct else 0,
            "points": points_earned,
            "completed": 1 >= total_questions,
        }).execute()

    return templates.TemplateResponse("quiz_result.html", _ctx(
        request,
        paper=paper,
        question=question,
        selected=selected,
        is_correct=is_correct,
        points_earned=points_earned,
    ))
