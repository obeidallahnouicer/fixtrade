import psycopg2

conn = psycopg2.connect("postgresql://postgres:@localhost:5432/fixtrade")
cur = conn.cursor()

print("=== known_anomalies ===")
cur.execute("SELECT COUNT(*) FROM known_anomalies")
print(f"Total: {cur.fetchone()[0]}")
cur.execute("SELECT symbol, COUNT(*) FROM known_anomalies GROUP BY symbol ORDER BY symbol")
for r in cur.fetchall():
    print(f"  {r[0]:20s} → {r[1]}")

print("\n=== intraday_ticks ===")
cur.execute("SELECT COUNT(*) FROM intraday_ticks")
print(f"Total: {cur.fetchone()[0]}")
cur.execute("SELECT symbol, COUNT(*), MIN(tick_timestamp)::date, MAX(tick_timestamp)::date FROM intraday_ticks GROUP BY symbol ORDER BY symbol")
for r in cur.fetchall():
    print(f"  {r[0]:20s} → {r[1]:>8,} ticks  ({r[2]} to {r[3]})")

print("\n=== Sample ticks (BIAT, first 5) ===")
cur.execute("SELECT tick_timestamp, price, volume FROM intraday_ticks WHERE symbol = 'BIAT' ORDER BY tick_timestamp LIMIT 5")
for r in cur.fetchall():
    print(f"  {r[0]}  price={r[1]}  vol={r[2]}")

conn.close()
