"""Post-ETL database verification."""
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
print("=== Table Row Counts ===")
for (t,) in cur.fetchall():
    cur.execute(f"SELECT COUNT(*) FROM {t}")
    (cnt,) = cur.fetchone()
    print(f"  {t}: {cnt:,} rows")

# Sample watermarks
print("\n=== ETL Watermarks (sample) ===")
cur.execute("SELECT layer, ticker, last_date, rows_processed FROM etl_watermarks WHERE ticker = '__global__' OR ticker IN ('BIAT', 'BH', 'SFBT') ORDER BY layer, ticker LIMIT 10")
for row in cur.fetchall():
    print(f"  {row}")

# Sample stock_prices
print("\n=== stock_prices (sample, last 5 by date) ===")
cur.execute("SELECT symbol, seance, cloture, quantite_negociee FROM stock_prices ORDER BY seance DESC LIMIT 5")
for row in cur.fetchall():
    print(f"  {row}")

# Count unique symbols
cur.execute("SELECT COUNT(DISTINCT symbol) FROM stock_prices")
(n_symbols,) = cur.fetchone()
print(f"\n=== Unique symbols in stock_prices: {n_symbols} ===")

# Watermark count
cur.execute("SELECT COUNT(*) FROM etl_watermarks")
(n_wm,) = cur.fetchone()
print(f"=== Total watermarks: {n_wm} ===")

conn.close()
print("\nDone!")
