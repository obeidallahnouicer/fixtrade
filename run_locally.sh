#!/bin/bash
# Local development runner - starts app with all services from Docker

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 FixTrade Local Development Startup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if Docker containers are running
echo "📦 Checking Docker services..."
docker compose -f docker-compose.local.yml ps

echo ""
echo "✅ Services ready:"
echo "   • PostgreSQL:  localhost:5432"
echo "   • pgAdmin:     http://localhost:5050"
echo "   • Redis:       localhost:6379"
echo ""

# Install dependencies if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📥 Installing Python dependencies..."
    pip install -q -r requirements.txt --timeout 120 || {
        echo "⚠️  Some packages may have failed to install due to network issues"
        echo "   Installing critical packages only..."
        pip install -q fastapi uvicorn sqlalchemy psycopg2-binary redis
    }
fi

echo ""
echo "🎯 Starting FastAPI app..."
echo "   Available at: http://localhost:8000"
echo "   Docs at:      http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "$(dirname "$0")"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
