from collections import OrderedDict

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.auth import clear_auth_cookies, get_current_user, set_auth_cookies
from app.db import get_admin_client, get_client

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

    import json
    admin = get_admin_client()
    paper = get_client().table("papers").select("*").eq("slug", slug).single().execute().data
    questions = (
        get_client().table("questions").select("*")
        .eq("paper_number", paper["number"]).execute().data
    )

    if not questions:
        return templates.TemplateResponse("quiz.html", _ctx(
            request, paper=paper, question=None, total=0, current=0,
        ))

    # Find first unanswered question for this user (admin to bypass RLS)
    attempts = (
        admin.table("quiz_attempts").select("question_id")
        .eq("user_id", user["id"]).execute().data
    )
    answered_ids = set(a["question_id"] for a in attempts)
    unanswered = [q for q in questions if q["id"] not in answered_ids]

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

    import json
    db = get_client()
    admin = get_admin_client()
    paper = db.table("papers").select("*").eq("slug", slug).single().execute().data

    question = db.table("questions").select("*").eq("id", question_id).single().execute().data
    if isinstance(question["choices"], str):
        question["choices"] = json.loads(question["choices"])

    is_correct = selected == question["correct_index"]
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
        admin.table("questions").select("id")
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


# --- Dashboard ---

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    admin = get_admin_client()
    db = get_client()

    # Get user's progress across all papers
    progress = (
        admin.table("user_paper_progress")
        .select("*")
        .eq("user_id", user["id"])
        .order("paper_number")
        .execute().data
    )

    # Get paper titles for display
    paper_numbers = [p["paper_number"] for p in progress]
    papers_map = {}
    if paper_numbers:
        papers = db.table("papers").select("number, title, slug").in_("number", paper_numbers).execute().data
        papers_map = {p["number"]: p for p in papers}

    # Attach paper info to progress
    for p in progress:
        p["paper"] = papers_map.get(p["paper_number"], {})

    # Get user profile
    profile = admin.table("profiles").select("*").eq("id", user["id"]).execute().data
    profile = profile[0] if profile else {"username": user.get("email", ""), "display_name": "Scholar"}

    # Stats
    total_points = sum(p["points"] for p in progress)
    papers_completed = sum(1 for p in progress if p["completed"])
    total_correct = sum(p["questions_correct"] for p in progress)
    total_answered = sum(p["questions_answered"] for p in progress)
    accuracy = round(total_correct / total_answered * 100) if total_answered else 0

    return templates.TemplateResponse("dashboard.html", _ctx(
        request,
        profile=profile,
        progress=progress,
        total_points=total_points,
        papers_completed=papers_completed,
        total_correct=total_correct,
        total_answered=total_answered,
        accuracy=accuracy,
    ))


# --- Leaderboard ---

@app.get("/leaderboard", response_class=HTMLResponse)
async def leaderboard(request: Request):
    admin = get_admin_client()

    # Aggregate points per user
    all_progress = admin.table("user_paper_progress").select("user_id, points, questions_correct, questions_answered, completed").execute().data

    # Group by user
    user_stats: dict[str, dict] = {}
    for p in all_progress:
        uid = p["user_id"]
        if uid not in user_stats:
            user_stats[uid] = {"points": 0, "correct": 0, "answered": 0, "completed": 0}
        user_stats[uid]["points"] += p["points"]
        user_stats[uid]["correct"] += p["questions_correct"]
        user_stats[uid]["answered"] += p["questions_answered"]
        user_stats[uid]["completed"] += 1 if p["completed"] else 0

    # Get profiles for all users
    if user_stats:
        profiles = admin.table("profiles").select("id, username, display_name").in_("id", list(user_stats.keys())).execute().data
        profiles_map = {p["id"]: p for p in profiles}
    else:
        profiles_map = {}

    # Build ranked list
    rankings = []
    for uid, stats in user_stats.items():
        profile = profiles_map.get(uid, {"username": "unknown", "display_name": "Unknown"})
        accuracy = round(stats["correct"] / stats["answered"] * 100) if stats["answered"] else 0
        rankings.append({
            "username": profile.get("display_name") or profile.get("username", "Unknown"),
            "points": stats["points"],
            "papers_completed": stats["completed"],
            "accuracy": accuracy,
            "total_answered": stats["answered"],
        })

    rankings.sort(key=lambda x: x["points"], reverse=True)

    return templates.TemplateResponse("leaderboard.html", _ctx(
        request,
        rankings=rankings,
    ))
