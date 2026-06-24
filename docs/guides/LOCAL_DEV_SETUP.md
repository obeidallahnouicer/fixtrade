# FixTrade Local Development Setup

## ✅ What's Running

You now have a **hybrid setup**: 
- **Services in Docker**: PostgreSQL, pgAdmin, Redis (lightweight, no build issues)
- **App locally**: FastAPI app runs on your machine (avoids xgboost build timeout)

### Running Services

```bash
# Status of Docker services
docker compose -f docker-compose.local.yml ps

# Stop services (when done)
docker compose -f docker-compose.local.yml down

# Restart services
docker compose -f docker-compose.local.yml restart
```

## 🚀 Quick Start

### Option 1: Use the startup script (Recommended)
```bash
./run_locally.sh
```
This will:
1. Check Docker services are running
2. Install Python dependencies if needed
3. Start FastAPI with hot reload at `http://localhost:8000`

### Option 2: Manual startup
```bash
# Start Docker services
docker compose -f docker-compose.local.yml up -d

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn app.main:app --reload
```

## 🛠️ Database Management

### Access pgAdmin (UI)
- **URL**: http://localhost:5050
- **Email**: `admin@fixtrade.local`
- **Password**: `admin`

### Load Initial Data
```bash
# Load stock price history (810k rows)
python3 db/load_data.py

# Generate intraday ticks & load anomalies
python3 db/load_intraday_and_labels.py --symbols BIAT SFBT BT --days 30

# Load all with single-minute resolution (testing)
python3 db/load_intraday_and_labels.py --symbols BIAT --days 5
```

### Direct Database Access (if psql installed)
```bash
psql -h localhost -U fixtrade -d fixtrade
```

Or use Python:
```bash
python3 -c "
import psycopg2
conn = psycopg2.connect('dbname=fixtrade user=fixtrade password=fixtrade host=localhost')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM stock_prices;')
print('Total price records:', cur.fetchone()[0])
cur.close()
conn.close()
"
```

## 🔧 Environment Variables

Your `.env` is configured for localhost:
```properties
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=fixtrade
POSTGRES_USER=fixtrade
POSTGRES_PASSWORD=fixtrade
REDIS_HOST=localhost
REDIS_PORT=6379
DATABASE_URL=postgresql://fixtrade:fixtrade@localhost:5432/fixtrade
```

## 📊 Available Endpoints

Once app is running at `http://localhost:8000`:

- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health

## 🐍 Run Scripts

### Sentiment Analysis
```bash
python3 scripts/analyze_sentiment.py --batch-size 50
```

### Data Verification
```bash
python3 _check_db.py
python3 _check_tables.py
python3 _verify_etl_db.py
```

### Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```

## 🐛 Troubleshooting

### PostgreSQL Connection Issues
```bash
# Test connection
python3 -c "
import psycopg2
try:
    psycopg2.connect('dbname=fixtrade user=fixtrade password=fixtrade host=localhost')
    print('✅ Connected!')
except Exception as e:
    print(f'❌ Error: {e}')
"

# If failed, restart containers
docker compose -f docker-compose.local.yml restart postgres
```

### Redis Connection Issues
```bash
# Test Redis
python3 -c "
import redis
r = redis.Redis(host='localhost', port=6379, db=0)
print(r.ping())
"
```

### Port Already in Use
```bash
# Find what's using port 5432 (PostgreSQL)
lsof -i :5432

# Or use different port in docker-compose.local.yml
# Change: 5432:5432 → 5433:5432
```

### Out of Memory
If you hit memory issues:
```bash
# Check disk space
df -h

# Reduce Redis memory in docker-compose.local.yml
# Change: --maxmemory 256mb → --maxmemory 128mb
```

## 📝 Notes

- **No Docker build**: Avoids xgboost timeout issues
- **Hot reload**: FastAPI auto-restarts on code changes
- **pgAdmin UI**: Great for exploring data visually
- **Idempotent scripts**: Safe to run multiple times (uses `ON CONFLICT DO NOTHING`)

## 🎯 Next Steps

1. **Access pgAdmin**: Visit http://localhost:5050
2. **Load data**: Run `python3 db/load_data.py`
3. **Start app**: Run `./run_locally.sh`
4. **Check API**: Visit http://localhost:8000/docs
5. **Run sentiment analysis**: `python3 scripts/analyze_sentiment.py`

---

**Last Updated**: 2026-05-25  
**Setup Type**: Local Python + Docker Services  
**No Docker app builds needed** ✅
