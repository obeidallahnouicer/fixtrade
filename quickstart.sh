#!/bin/bash
# FixTrade — Quick Start Script
# Helps you train models and run predictions

set -e  # Exit on error

cd "$(dirname "$0")"

echo "🚀 FixTrade Quick Start"
echo "======================="
echo ""

# Activate venv
if [ -d ".venv" ]; then
    echo "✅ Activating virtual environment..."
    source .venv/bin/activate
else
    echo "❌ Virtual environment not found. Run: python -m venv .venv"
    exit 1
fi

# Check what action to perform
case "${1:-help}" in
    train)
        echo "🎓 Training prediction models..."
        echo ""
        python -m prediction train
        ;;
    
    predict)
        SYMBOL="${2:-BIAT}"
        echo "🔮 Running prediction for $SYMBOL..."
        echo ""
        python -m prediction predict --symbol "$SYMBOL" --days 5
        ;;
    
    backend)
        echo "🚀 Starting FastAPI backend on port 8000..."
        echo ""
        uvicorn app.main:app --reload --port 8000
        ;;
    
    frontend)
        echo "🎨 Starting Streamlit dashboard on port 8501..."
        echo ""
        streamlit run streamlit_app.py --server.port 8501
        ;;
    
    both)
        echo "🚀 Starting both backend and frontend..."
        echo ""
        echo "Starting FastAPI in background..."
        uvicorn app.main:app --port 8000 &
        BACKEND_PID=$!
        sleep 3
        echo "✅ Backend started (PID: $BACKEND_PID)"
        echo ""
        echo "Starting Streamlit..."
        streamlit run streamlit_app.py --server.port 8501
        ;;
    
    etl)
        echo "🔄 Running ETL pipeline..."
        echo ""
        python -m prediction.etl.pipeline
        ;;
    
    db-check)
        echo "🔍 Checking database connection..."
        echo ""
        python scripts/check_db_connect.py
        ;;
    
    help|*)
        echo "Usage: ./quickstart.sh [command]"
        echo ""
        echo "Commands:"
        echo "  train       Train prediction models"
        echo "  predict     Run prediction for a symbol (default: BIAT)"
        echo "              Usage: ./quickstart.sh predict SYMBOL"
        echo "  backend     Start FastAPI backend"
        echo "  frontend    Start Streamlit dashboard"
        echo "  both        Start both backend and frontend"
        echo "  etl         Run ETL pipeline"
        echo "  db-check    Check database connection"
        echo "  help        Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./quickstart.sh train"
        echo "  ./quickstart.sh predict BIAT"
        echo "  ./quickstart.sh both"
        echo ""
        ;;
esac
