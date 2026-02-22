#!/usr/bin/env python3
"""Generate quiz questions from paper metadata and abstracts.

Uses template-based generation — no API costs.
For higher quality, set ANTHROPIC_API_KEY and use --ai flag.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_admin_client


def generate_template_questions(paper: dict, all_papers: list[dict]) -> list[dict]:
    """Generate simple quiz questions from paper metadata."""
    questions = []
    p = paper
    others = [o for o in all_papers if o["number"] != p["number"] and o["category"] == p["category"]]

    # Q1: Year question
    if p["year"]:
        wrong_years = [y for y in [2020, 2021, 2022, 2023, 2024, 2025] if y != p["year"]]
        random.shuffle(wrong_years)
        choices = [str(p["year"])] + [str(y) for y in wrong_years[:3]]
        random.shuffle(choices)
        questions.append({
            "paper_number": p["number"],
            "question": f"In what year was \"{p['title']}\" published?",
            "choices": choices,
            "correct_index": choices.index(str(p["year"])),
            "explanation": f"This paper was published in {p['year']}.",
        })

    # Q2: Authors question
    if p["authors"] and others:
        wrong_authors = [o["authors"] for o in others if o["authors"]][:3]
        if len(wrong_authors) >= 3:
            choices = [p["authors"]] + wrong_authors
            random.shuffle(choices)
            questions.append({
                "paper_number": p["number"],
                "question": f"Who are the authors of \"{p['title']}\"?",
                "choices": choices,
                "correct_index": choices.index(p["authors"]),
                "explanation": f"The authors are {p['authors']}.",
            })

    # Q3: Category question
    if p["category"]:
        all_cats = list(set(o["category"] for o in all_papers))
        wrong_cats = [c for c in all_cats if c != p["category"]]
        random.shuffle(wrong_cats)
        if len(wrong_cats) >= 3:
            choices = [p["category"]] + wrong_cats[:3]
            random.shuffle(choices)
            questions.append({
                "paper_number": p["number"],
                "question": f"Which category does \"{p['title']}\" belong to?",
                "choices": choices,
                "correct_index": choices.index(p["category"]),
                "explanation": f"This paper belongs to the \"{p['category']}\" category.",
            })

    return questions


def generate_ai_questions(paper: dict) -> list[dict]:
    """Generate quiz questions using Claude API. Requires ANTHROPIC_API_KEY."""
    import os
    try:
        import anthropic
    except ImportError:
        print("  Install anthropic: pip install anthropic")
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  Set ANTHROPIC_API_KEY to use AI generation")
        return []

    if not paper.get("summary"):
        return []

    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""Based on this robotics paper, generate 3 multiple-choice quiz questions.

Title: {paper['title']}
Authors: {paper.get('authors', 'Unknown')}
Year: {paper.get('year', 'Unknown')}
Category: {paper['category']}
Abstract: {paper['summary']}

Return valid JSON array. Each item must have:
- "question": the question text
- "choices": array of exactly 4 answer strings
- "correct_index": index (0-3) of the correct answer
- "explanation": brief explanation of the correct answer

Focus on testing understanding of the paper's key contributions and methods.
Return ONLY the JSON array, no other text."""

    resp = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        text = resp.content[0].text.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        items = json.loads(text)
        for item in items:
            item["paper_number"] = paper["number"]
        return items
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  Failed to parse AI response: {e}")
        return []


def main():
    use_ai = "--ai" in sys.argv
    random.seed(42)  # Reproducible for template mode

    client = get_admin_client()
    papers = client.table("papers").select("*").order("number").execute().data

    # Check existing questions
    existing = client.table("questions").select("paper_number").execute().data
    existing_numbers = set(q["paper_number"] for q in existing)

    total = 0
    for paper in papers:
        if paper["number"] in existing_numbers:
            continue

        if use_ai:
            questions = generate_ai_questions(paper)
        else:
            questions = generate_template_questions(paper, papers)

        if questions:
            # Convert choices to JSON strings for storage
            for q in questions:
                q["choices"] = json.dumps(q["choices"])

            client.table("questions").insert(questions).execute()
            total += len(questions)
            print(f"  #{paper['number']}: {len(questions)} questions")

    print(f"\nGenerated {total} total questions ({'AI' if use_ai else 'template'} mode)")


if __name__ == "__main__":
    main()
