#!/bin/bash
# Quick start script for FixTrade development

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

echo "========================================"
echo "  FixTrade - Development Environment"
echo "========================================"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ ERROR: .env file not found!"
    echo "   Please create .env with DATABASE_URL set"
    exit 1
fi

# Check if Docker services are running
echo "🔍 Checking Docker services..."
if ! docker compose -f docker-compose.local.yml ps | grep -q "running"; then
    echo "⚠️  Some Docker services are not running"
    echo "   Run this first: docker compose -f docker-compose.local.yml up -d"
    echo ""
fi

# Kill any existing processes on port 8000
if lsof -i :8000 >/dev/null 2>&1; then
    echo "🧹 Cleaning up port 8000..."
    lsof -ti :8000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated"
    echo "   Run: source .venv/bin/activate"
    echo ""
fi

# Start the app
echo ""
echo "✨ Starting FixTrade FastAPI server..."
echo "   📍 URL: http://127.0.0.1:8000"
echo "   📚 Docs: http://127.0.0.1:8000/docs"
echo "   🔴 Stop: Press Ctrl+C"
echo ""

exec python3 run_app.py
