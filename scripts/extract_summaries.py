#!/usr/bin/env python3
"""Extract abstracts from PDFs and store as summaries in Supabase. No API costs."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_admin_client

PAPERS_DIR = Path("/Users/bot_user/robotics-papers")


def extract_abstract(pdf_path: Path) -> str | None:
    """Extract the abstract section from a PDF."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  Could not open {pdf_path.name}: {e}")
        return None

    # Get text from first 2 pages (abstract is always early)
    text = ""
    for page_num in range(min(2, len(doc))):
        text += doc[page_num].get_text()
    doc.close()

    if not text.strip():
        return None

    # Try to find abstract section
    # Common patterns: "Abstract", "ABSTRACT", "Abstract.", "Abstract—"
    abstract_match = re.search(
        r"(?:^|\n)\s*(?:ABSTRACT|Abstract)[.:\s—\-]*\n?(.*?)(?:\n\s*(?:1[\s.]|I[\s.]|Introduction|INTRODUCTION|Keywords|Index Terms|CCS Concepts))",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if abstract_match:
        abstract = abstract_match.group(1).strip()
    else:
        # Fallback: grab text between "Abstract" and next section header
        parts = re.split(r"(?:ABSTRACT|Abstract)[.:\s—\-]*", text, maxsplit=1)
        if len(parts) > 1:
            # Take first ~2000 chars after "Abstract" and cut at likely section break
            chunk = parts[1][:2000]
            # Cut at first numbered section or common headers
            cut = re.search(r"\n\s*(?:\d+[\s.]|I+[\s.])\s*[A-Z]", chunk)
            abstract = chunk[:cut.start()].strip() if cut else chunk.strip()
        else:
            return None

    # Clean up: collapse whitespace, remove hyphenation at line breaks
    abstract = re.sub(r"-\n", "", abstract)
    abstract = re.sub(r"\n", " ", abstract)
    abstract = re.sub(r"\s+", " ", abstract)

    # Skip if too short (probably failed extraction)
    if len(abstract) < 50:
        return None

    return abstract


def find_pdf(pdf_filename: str) -> Path | None:
    """Find a PDF in the category subdirectories."""
    for subdir in PAPERS_DIR.iterdir():
        if subdir.is_dir():
            candidate = subdir / pdf_filename
            if candidate.exists():
                return candidate
    return None


def main():
    client = get_admin_client()
    result = client.table("papers").select("number, slug, title, pdf_filename, summary").order("number").execute()

    updated = 0
    skipped = 0
    failed = 0

    for paper in result.data:
        if paper["summary"]:
            skipped += 1
            continue

        if not paper["pdf_filename"]:
            continue

        pdf_path = find_pdf(paper["pdf_filename"])
        if not pdf_path:
            print(f"  #{paper['number']}: PDF not found: {paper['pdf_filename']}")
            failed += 1
            continue

        abstract = extract_abstract(pdf_path)
        if not abstract:
            print(f"  #{paper['number']}: Could not extract abstract from {paper['pdf_filename']}")
            failed += 1
            continue

        client.table("papers").update({"summary": abstract}).eq("number", paper["number"]).execute()
        print(f"  #{paper['number']}: {paper['title'][:50]}... ({len(abstract)} chars)")
        updated += 1

    print(f"\nDone: {updated} updated, {skipped} already had summaries, {failed} failed")


if __name__ == "__main__":
    main()
