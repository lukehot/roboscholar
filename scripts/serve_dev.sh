#!/usr/bin/env bash
# Run the app for local + same-WiFi phone access (bind to all interfaces).
set -e
cd "$(dirname "$0")/.."
PORT="${PORT:-8000}"
echo "Starting RoboScholar on http://0.0.0.0:${PORT}"
echo "Same WiFi: use your machine IP, e.g. http://$(ipconfig getifaddr en0 2>/dev/null || hostname):${PORT}"
exec uv run uvicorn app.main:app --reload --host 0.0.0.0 --port "$PORT"
