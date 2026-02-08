"""Quick check of database tables."""
import pg8000

conn = pg8000.connect(host="localhost", port=5432, database="fixtrade", user="postgres", password="")
cur = conn.cursor()

# Check stock_prices columns and sample
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='stock_prices' ORDER BY ordinal_position")
print("stock_prices columns:")
for c in cur.fetchall():
    print(f"  {c[0]}: {c[1]}")

cur.execute("SELECT * FROM stock_prices LIMIT 2")
print("\nSample rows:", cur.fetchall())

# Check price_predictions columns
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='price_predictions' ORDER BY ordinal_position")
print("\nprice_predictions columns:")
for c in cur.fetchall():
    print(f"  {c[0]}: {c[1]}")

# Check etl_watermarks columns
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='etl_watermarks' ORDER BY ordinal_position")
print("\netl_watermarks columns:")
for c in cur.fetchall():
    print(f"  {c[0]}: {c[1]}")

# Check model_registry columns
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='model_registry' ORDER BY ordinal_position")
print("\nmodel_registry columns:")
for c in cur.fetchall():
    print(f"  {c[0]}: {c[1]}")

# Check if there are volume/liquidity prediction tables
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name LIKE '%volume%' OR table_name LIKE '%liquid%'")
print("\nVolume/Liquidity tables:", [r[0] for r in cur.fetchall()])

conn.close()
