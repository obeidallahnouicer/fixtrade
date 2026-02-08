"""Quick check of all tables in the fixtrade database."""
import pg8000

conn = pg8000.connect(host="localhost", port=5432, database="fixtrade", user="postgres", password="")
cur = conn.cursor()

# All tables
cur.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name
""")
tables = [r[0] for r in cur.fetchall()]

print("=" * 60)
print(f"{'TABLE':35s} {'ROWS':>12s}")
print("=" * 60)

for t in tables:
    cur.execute(f"SELECT COUNT(*) FROM {t}")
    count = cur.fetchone()[0]
    print(f"  {t:33s} {count:>10,}")

print("=" * 60)

# Views
cur.execute("""
    SELECT table_name
    FROM information_schema.views
    WHERE table_schema = 'public'
    ORDER BY table_name
""")
views = [r[0] for r in cur.fetchall()]
if views:
    print(f"\n{'VIEW':35s} {'ROWS':>12s}")
    print("-" * 60)
    for v in views:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {v}")
            count = cur.fetchone()[0]
            print(f"  {v:33s} {count:>10,}")
        except Exception as e:
            print(f"  {v:33s} {'ERROR':>12s}")

# Sample data from prediction/model tables
for table in ["price_predictions", "volume_predictions", "liquidity_predictions", "model_registry"]:
    if table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        if count > 0:
            print(f"\n--- Sample from {table} (last 5) ---")
            cur.execute(f"SELECT * FROM {table} ORDER BY created_at DESC LIMIT 5")
            cols = [desc[0] for desc in cur.description]
            print("  " + " | ".join(cols))
            for row in cur.fetchall():
                print("  " + " | ".join(str(v) for v in row))
        else:
            print(f"\n--- {table}: EMPTY (0 rows) ---")

# ETL watermarks sample
if "etl_watermarks" in tables:
    cur.execute("SELECT COUNT(*) FROM etl_watermarks")
    wm_count = cur.fetchone()[0]
    print(f"\n--- etl_watermarks: {wm_count} rows ---")
    cur.execute("SELECT * FROM etl_watermarks ORDER BY updated_at DESC LIMIT 10")
    cols = [desc[0] for desc in cur.description]
    print("  " + " | ".join(cols))
    for row in cur.fetchall():
        print("  " + " | ".join(str(v) for v in row))

# Stock prices summary
if "stock_prices" in tables:
    cur.execute("SELECT COUNT(*) FROM stock_prices")
    sp_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT symbol) FROM stock_prices")
    sym_count = cur.fetchone()[0]
    cur.execute("SELECT MIN(seance), MAX(seance) FROM stock_prices")
    min_d, max_d = cur.fetchone()
    print(f"\n--- stock_prices: {sp_count:,} rows, {sym_count} symbols, {min_d} → {max_d} ---")

conn.close()
