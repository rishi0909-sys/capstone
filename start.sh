#!/usr/bin/env bash

# ==============================================================================
# Unified Startup Script for Intelligent Pattern Discovery Engine
# Starts:
#   1. FastAPI Backend (http://127.0.0.1:8000)
#   2. Vite / Next-style Frontend (http://127.0.0.1:3000)
#
# Press Ctrl+C at any time to cleanly stop both servers.
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

echo "========================================================"
echo "🚀 Starting Intelligent Pattern Discovery Engine"
echo "========================================================"

# Function to cleanly kill background child processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down all servers..."
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    # Ensure port cleanup
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    echo "✅ Both servers stopped successfully."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# 1. Start FastAPI Backend
echo "📦 [1/2] Starting FastAPI Backend on http://127.0.0.1:8000..."
cd "$BACKEND_DIR"

if [ -f "venv/bin/uvicorn" ]; then
    PYTHON_EXEC="venv/bin/uvicorn"
elif command -v uvicorn &> /dev/null; then
    PYTHON_EXEC="uvicorn"
else
    echo "❌ Error: uvicorn not found in backend/venv or PATH. Please activate your environment."
    exit 1
fi

$PYTHON_EXEC main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

# Brief wait to verify backend port binding
sleep 1.5

# 2. Start Vite Frontend
echo "💻 [2/2] Starting Vite Frontend on http://127.0.0.1:3000..."
cd "$FRONTEND_DIR"

if [ ! -d "node_modules" ]; then
    echo "⚠️ node_modules missing in frontend. Running npm install..."
    npm install
fi

./node_modules/.bin/vite --port 3000 --host 127.0.0.1 &
FRONTEND_PID=$!

echo ""
echo "========================================================"
echo "✨ All services running!"
echo "   🌐 Frontend UI:    http://127.0.0.1:3000"
echo "   ⚙️  Backend API:    http://127.0.0.1:8000"
echo "   📖 API Docs:       http://127.0.0.1:8000/docs"
echo "========================================================"
echo "💡 Press [Ctrl + C] in this window to stop both servers."
echo ""

# Keep script running and wait for background jobs
wait "$BACKEND_PID" "$FRONTEND_PID"
