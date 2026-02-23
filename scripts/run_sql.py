#!/usr/bin/env python3
"""Run SQL against Supabase Postgres directly."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def run_sql(sql: str) -> None:
    db_url = os.environ["DATABASE_URL"].split("?")[0]  # Strip query params
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(sql)
    try:
        rows = cur.fetchall()
        for row in rows:
            print(row)
    except psycopg2.ProgrammingError:
        pass  # No results (DDL statement)
    cur.close()
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run SQL from file
        sql = Path(sys.argv[1]).read_text()
    else:
        sql = sys.stdin.read()
    run_sql(sql)
    print("OK")
