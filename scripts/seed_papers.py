#!/usr/bin/env python3
"""Parse README.md and seed papers into Supabase."""

import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_admin_client


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def parse_readme(readme_path: str) -> list[dict]:
    text = Path(readme_path).read_text()
    papers = []
    current_category = ""

    for line in text.split("\n"):
        # Match category headers like "## 1. Foundational VLA Papers"
        cat_match = re.match(r"^## \d+\.\s+(.+)$", line)
        if cat_match:
            current_category = cat_match.group(1).strip()
            continue

        # Match table rows: | # | File | Paper | Authors | Year | Link |
        row_match = re.match(
            r"\|\s*(\d+)\s*\|"   # number
            r"\s*(.*?)\s*\|"      # file
            r"\s*(.*?)\s*\|"      # paper title
            r"\s*(.*?)\s*\|"      # authors
            r"\s*(.*?)\s*\|"      # year
            r"\s*(.*?)\s*\|",     # link
            line,
        )
        if not row_match:
            continue

        number = int(row_match.group(1))
        file_col = row_match.group(2).strip()
        title = row_match.group(3).strip()
        authors = row_match.group(4).strip() or None
        year_str = row_match.group(5).strip()
        link_col = row_match.group(6).strip()

        # Extract PDF filename from backtick-wrapped text
        pdf_match = re.search(r"`(.+?\.pdf)`", file_col)
        pdf_filename = pdf_match.group(1) if pdf_match else None

        # Extract URL from markdown link
        link_match = re.search(r"\[.*?\]\((.*?)\)", link_col)
        link = link_match.group(1) if link_match else None

        # Parse year (handle ranges like "2024-2025")
        year_match = re.search(r"(\d{4})", year_str)
        year = int(year_match.group(1)) if year_match else None

        slug = slugify(title)

        papers.append({
            "number": number,
            "slug": slug,
            "title": title,
            "authors": authors,
            "year": year,
            "category": current_category,
            "pdf_filename": pdf_filename,
            "link": link,
        })

    return papers


# Also handle section 15 (Textbooks) which has a different table format
def parse_textbooks(readme_path: str) -> list[dict]:
    text = Path(readme_path).read_text()
    papers = []
    in_textbooks = False

    for line in text.split("\n"):
        if "## 15. Textbooks" in line:
            in_textbooks = True
            continue
        if in_textbooks and line.startswith("## "):
            break
        if not in_textbooks:
            continue

        # | # | Title | Authors | Link |
        row_match = re.match(
            r"\|\s*(\d+)\s*\|"
            r"\s*(.*?)\s*\|"
            r"\s*(.*?)\s*\|"
            r"\s*(.*?)\s*\|",
            line,
        )
        if not row_match:
            continue

        number = int(row_match.group(1))
        title = row_match.group(2).strip()
        authors = row_match.group(3).strip() or None
        link_col = row_match.group(4).strip()

        link_match = re.search(r"\[.*?\]\((.*?)\)", link_col)
        link = link_match.group(1) if link_match else None

        slug = slugify(title)

        papers.append({
            "number": number,
            "slug": slug,
            "title": title,
            "authors": authors,
            "year": None,
            "category": "Textbooks & Classic References",
            "pdf_filename": None,
            "link": link,
        })

    return papers


def main():
    readme = "/Users/bot_user/robotics-papers/README.md"
    papers = parse_readme(readme)
    textbooks = parse_textbooks(readme)
    all_papers = papers + textbooks

    # Deduplicate by number (textbooks may already be in main parse)
    seen = set()
    unique = []
    for p in all_papers:
        if p["number"] not in seen:
            seen.add(p["number"])
            unique.append(p)

    print(f"Parsed {len(unique)} papers")
    for p in unique[:3]:
        print(f"  #{p['number']}: {p['title'][:60]}...")

    client = get_admin_client()

    # Upsert papers
    result = client.table("papers").upsert(unique, on_conflict="number").execute()
    print(f"Upserted {len(result.data)} papers into Supabase")


if __name__ == "__main__":
    main()
