#!/usr/bin/env bash
# Run the app and expose it via ngrok so you can open it from your phone anywhere.
# Requires: ngrok (brew install ngrok) and ngrok auth if needed.
set -e
cd "$(dirname "$0")/.."
PORT="${PORT:-8000}"

if ! command -v ngrok &>/dev/null; then
  echo "ngrok not found. Install with: brew install ngrok"
  exit 1
fi

# Start uvicorn in background
uv run uvicorn app.main:app --host 127.0.0.1 --port "$PORT" &
UVICORN_PID=$!
trap 'kill $UVICORN_PID 2>/dev/null' EXIT

# Wait for server to be up
for i in {1..15}; do
  curl -s -o /dev/null "http://127.0.0.1:${PORT}/" && break
  sleep 1
done

echo "App running. Starting tunnel..."
echo "Open the HTTPS URL below on your phone."
exec ngrok http "$PORT"
