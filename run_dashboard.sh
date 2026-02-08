#!/bin/bash
# FixTrade — Launch Streamlit Dashboard
# Usage: ./run_dashboard.sh [--api-url http://localhost:8000/api/v1]

API_URL="${1:-http://localhost:8000/api/v1}"

echo "🚀 Starting FixTrade Streamlit Dashboard..."
echo "   API: $API_URL"
echo ""

FIXTRADE_API_URL="$API_URL" streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.headless true \
    --theme.base dark \
    --theme.primaryColor "#3b82f6" \
    --browser.gatherUsageStats false
