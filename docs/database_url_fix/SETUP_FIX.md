# 🎯 DATABASE_URL Fix - Complete Guide

## What Was Broken

When trying to access ANY endpoint that required database access, the app would crash with:

```
ValueError: DATABASE_URL environment variable not set
```

Affected endpoints:
- ❌ `POST /api/v1/trading/anomalies` → 500 Error
- ❌ `POST /api/v1/trading/sentiment` → 500 Error  
- ❌ `POST /api/v1/trading/predictions` → 500 Error
- And all other trading endpoints

## What Was The Problem

1. The `.env` file had `DATABASE_URL` configured correctly
2. But when running `uvicorn app.main:app`, the `.env` file was NOT being loaded
3. Repository adapters use `os.getenv("DATABASE_URL")` at initialization time
4. This happens BEFORE Pydantic's settings are loaded
5. Result: `os.getenv()` returns `None` and crashes with `ValueError`

## The Fix

### 1. Use the New Launcher Script

**Stop using:**
```bash
uvicorn app.main:app --reload
```

**Start using:**
```bash
python3 run_app.py
```

Or for convenience:
```bash
./start_dev.sh
```

### 2. What the Fix Does

The `run_app.py` script:
1. Loads `.env` **immediately** before any imports
2. Verifies `DATABASE_URL` is set
3. Starts uvicorn with proper configuration
4. Shows clear error messages if anything is wrong

## Quick Start

### Step 1: Ensure Docker Services Are Running

```bash
docker compose -f docker-compose.local.yml ps
```

You should see all 3 services running:
```
postgres   ... Up ... 5432
pgadmin    ... Up ... 5050
redis      ... Up ... 6379
```

If not, start them:
```bash
docker compose -f docker-compose.local.yml up -d
```

### Step 2: Activate Virtual Environment

```bash
source .venv/bin/activate
```

Or if using conda:
```bash
conda activate fixtrade
```

### Step 3: Start the App

```bash
python3 run_app.py
```

You should see:
```
✓ Loading environment from /home/obeid/Desktop/projects/fixtrade/.env
✓ DATABASE_URL is set: postgresql://fixtrade:fixtrade@localhost:5432/fixt...
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Started reloader process
```

### Step 4: Access the API

Open in browser or use curl:

```bash
# Health check
curl http://127.0.0.1:8000/api/v1/health

# Swagger UI documentation
open http://127.0.0.1:8000/docs

# Test an endpoint
curl -X POST http://127.0.0.1:8000/api/v1/trading/anomalies \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BIAT"}'
```

## Verification

### Before & After

**BEFORE (Broken)**
```
$ uvicorn app.main:app --reload
...
ValueError: DATABASE_URL environment variable not set
```

**AFTER (Fixed)**
```
$ python3 run_app.py
✓ Loading environment from /home/obeid/Desktop/projects/fixtrade/.env
✓ DATABASE_URL is set: postgresql://fixtrade:fixtrade@localhost:5432/fixtrade
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Test Endpoints

All these should now return **200 OK** (not 500):

```bash
# Health
curl http://127.0.0.1:8000/api/v1/health
→ 200 OK: {"status": "ok", "version": "1.0.0"}

# AI Status  
curl http://127.0.0.1:8000/api/v1/ai/status
→ 200 OK: {"module": "AI Decision Agent", ...}

# Anomalies (was 500, now works!)
curl -X POST http://127.0.0.1:8000/api/v1/trading/anomalies \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BIAT"}'
→ 200 OK: {"anomalies": []}

# Sentiment (was 500, now works!)
curl -X POST http://127.0.0.1:8000/api/v1/trading/sentiment \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BIAT", "text": "Great stock"}'
→ 200 OK: {"symbol": "BIAT", "date": "2026-05-25", ...}

# Swagger UI
curl -I http://127.0.0.1:8000/docs
→ 200 OK
```

## Files Created/Modified

### New Files
| File | Purpose |
|------|---------|
| `run_app.py` | ✨ **Main fix** - Loads .env before starting app |
| `start_dev.sh` | Convenience wrapper script (bash) |
| `DATABASE_URL_FIX.md` | Technical documentation of the fix |

### No Changes Needed
- ✅ `.env` - Already correct (DATABASE_URL set properly)
- ✅ `app/core/config.py` - Pydantic settings unchanged
- ✅ `app/infrastructure/trading/*.py` - Repository code unchanged
- ✅ `app/interfaces/trading/dependencies.py` - Unchanged

## Environment Variables

The app now correctly loads these from `.env`:

```
DATABASE_URL=postgresql://fixtrade:fixtrade@localhost:5432/fixtrade
POSTGRES_HOST=localhost
POSTGRES_USER=fixtrade
POSTGRES_PASSWORD=fixtrade
POSTGRES_DB=fixtrade
POSTGRES_PORT=5432
REDIS_HOST=redis
REDIS_PORT=6379
OPENROUTER_API_KEY=your_openrouter_api_key_here
DEFAULT_LLM_PROVIDER=openrouter
DEFAULT_LLM_MODEL=anthropic/claude-3.5-sonnet
... (and all others)
```

## Troubleshooting

### Problem: "Address already in use"
```
ERROR:    [Errno 98] Address already in use
```

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use the cleanup built into start_dev.sh
./start_dev.sh
```

### Problem: ".env file not found"
```
⚠ Warning: .env file not found at /path/to/fixtrade/.env
```

**Solution:**
```bash
# Check you're in the right directory
pwd
# Should output: /home/obeid/Desktop/projects/fixtrade

# Check .env exists
ls -la .env

# If missing, check git status
git status .env
```

### Problem: "DATABASE_URL not set after loading .env"
```
✗ ERROR: DATABASE_URL not set after loading .env
```

**Solution:**
```bash
# Check .env content
grep DATABASE_URL .env

# Should output:
# DATABASE_URL=postgresql://fixtrade:fixtrade@localhost:5432/fixtrade

# If empty/missing, edit .env and add it
```

### Problem: "Cannot connect to PostgreSQL"
```
WARNING | prediction.db_sink | Cannot connect to PostgreSQL
```

**Solution:**
```bash
# Check Docker services
docker compose -f docker-compose.local.yml ps

# Should show postgres running on port 5432
# If not, start them:
docker compose -f docker-compose.local.yml up -d

# Verify connection
psql postgresql://fixtrade:fixtrade@localhost:5432/fixtrade
```

## How It Works

### Before (Broken Flow)

```
$ uvicorn app.main:app --reload
│
├─ OS loads working directory
│
├─ .env file is NOT automatically loaded
│  (uvicorn doesn't load .env, only Pydantic does, and that happens later)
│
├─ FastAPI starts
│  └─ Request to /api/v1/trading/anomalies arrives
│     └─ Dependency injection runs
│        └─ get_detect_anomalies_use_case() called
│           └─ StockPriceRepositoryAdapter() initialized
│              └─ os.getenv("DATABASE_URL") returns None ❌
│                 └─ raise ValueError("DATABASE_URL environment variable not set")
│                    └─ 500 Internal Server Error ❌
```

### After (Fixed Flow)

```
$ python3 run_app.py
│
├─ run_app.py runs FIRST
│  └─ from dotenv import load_dotenv
│     └─ load_dotenv(".env", override=True)
│        └─ All variables from .env loaded into os.environ ✓
│           └─ os.environ["DATABASE_URL"] = "postgresql://..." ✓
│
├─ Uvicorn starts
│  └─ Request to /api/v1/trading/anomalies arrives
│     └─ Dependency injection runs
│        └─ get_detect_anomalies_use_case() called
│           └─ StockPriceRepositoryAdapter() initialized
│              └─ os.getenv("DATABASE_URL") returns "postgresql://..." ✓
│                 └─ Repository initialized successfully ✓
│                    └─ Handler returns 200 OK ✓
```

## Production Deployment

For production, you have these options:

### Option 1: Docker Container
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
COPY .env .env  # ← Make sure this is copied!
RUN pip install -r requirements.txt
CMD ["python3", "run_app.py"]
```

```bash
docker build -t fixtrade .
docker run -p 8000:8000 fixtrade
```

### Option 2: System Environment Variables
Instead of `.env` file, set environment variables:
```bash
export DATABASE_URL=postgresql://user:pass@host:5432/db
python3 run_app.py
```

The script works either way (`.env` file OR system env vars).

### Option 3: Use Gunicorn (Multi-worker)
```bash
# Still load .env, but use gunicorn for production
gunicorn -w 4 -b 0.0.0.0:8000 \
  --env DATABASE_URL=postgresql://... \
  app.main:app
```

## Summary

✅ **Issue**: DATABASE_URL not loaded at app startup  
✅ **Cause**: Missing explicit .env loading before app import  
✅ **Solution**: Use `python3 run_app.py` instead of `uvicorn app.main:app`  
✅ **Result**: All endpoints work, no more 500 errors  
✅ **Usage**: `python3 run_app.py` or `./start_dev.sh`  

---

**Need help?** Check the [DATABASE_URL_FIX.md](./DATABASE_URL_FIX.md) for technical details.
