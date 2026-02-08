"""Verify database tables and run DatabaseSink.ensure_tables()."""
import sys
sys.path.insert(0, ".")

from prediction.db_sink import DatabaseSink

sink = DatabaseSink()

# 1. Create missing tables (volume_predictions, liquidity_predictions)
print("=== Creating/verifying tables ===")
ok = sink.ensure_tables()
print(f"ensure_tables() -> {ok}")

# 2. Verify all tables exist now
import pg8000
conn = pg8000.connect(host="localhost", port=5432, database="fixtrade",
                      user="postgres", password="")
conn.autocommit = True
cur = conn.cursor()

cur.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
    ORDER BY table_name
""")
print("\n=== All tables ===")
for (t,) in cur.fetchall():
    cur.execute(f"SELECT COUNT(*) FROM {t}")
    (cnt,) = cur.fetchone()
    print(f"  {t}: {cnt} rows")

# 3. Quick smoke test: persist a watermark
from datetime import date
ok = sink.persist_watermark("bronze", "TEST_TICKER", date(2025, 1, 1), 42)
print(f"\nWatermark insert: {ok}")

cur.execute("SELECT * FROM etl_watermarks WHERE ticker = 'TEST_TICKER'")
rows = cur.fetchall()
print(f"Watermark rows: {rows}")

# Clean up test data
cur.execute("DELETE FROM etl_watermarks WHERE ticker = 'TEST_TICKER'")
print("Test row cleaned up.")

conn.close()
sink.close()
print("\nDone!")
