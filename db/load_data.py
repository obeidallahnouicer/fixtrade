"""
Load all BVMT historical data from CSV/TXT files into PostgreSQL.

Reads every file in data/raw/, parses the BVMT format, and inserts
into the stock_prices table using ON CONFLICT DO NOTHING (idempotent).

Usage:
    python db/load_data.py
"""

import csv
import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "fixtrade")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "")

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# ---------------------------------------------------------------------------
# BVMT column name mapping (French → normalized)
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "SEANCE": "seance",
    "GROUPE": "groupe",
    "CODE": "code_isin",
    "VALEUR": "symbol",
    "OUVERTURE": "ouverture",
    "CLOTURE": "cloture",
    "PLUS_BAS": "plus_bas",
    "PLUS_HAUT": "plus_haut",
    "QUANTITE_NEGOCIEE": "quantite_negociee",
    "NB_TRANSACTION": "nb_transaction",
    "CAPITAUX": "capitaux",
}


def detect_separator(first_line: str) -> str:
    """Detect if the file uses ; , tab, or fixed-width."""
    if ";" in first_line:
        return ";"
    if "," in first_line and first_line.count(",") > 3:
        return ","
    if "\t" in first_line:
        return "\t"
    # Fixed-width (the .txt files)
    return "FIXED"


def parse_fixed_width(lines: list[str]) -> list[dict]:
    """Parse BVMT fixed-width .txt format."""
    rows = []
    header_line = lines[0]

    # Skip the separator line (-------)
    data_start = 1
    for i, line in enumerate(lines[1:], 1):
        if line.strip().startswith("---"):
            data_start = i + 1
            break

    for line in lines[data_start:]:
        line = line.rstrip()
        if not line or line.startswith("---"):
            continue

        # Fixed-width positions based on BVMT format
        # Parse by splitting on 2+ spaces
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 10:
            continue

        try:
            row = {
                "seance": parts[0].strip(),
                "groupe": parts[1].strip(),
                "code_isin": parts[2].strip(),
                "symbol": parts[3].strip(),
                "ouverture": parts[4].strip().replace(",", "."),
                "cloture": parts[5].strip().replace(",", "."),
                "plus_bas": parts[6].strip().replace(",", "."),
                "plus_haut": parts[7].strip().replace(",", "."),
                "quantite_negociee": parts[8].strip(),
                "nb_transaction": parts[9].strip(),
            }
            if len(parts) > 10:
                row["capitaux"] = parts[10].strip().replace(",", ".")
            else:
                row["capitaux"] = "0"
            rows.append(row)
        except (IndexError, ValueError):
            continue
    return rows


def parse_csv(filepath: Path) -> list[dict]:
    """Parse a single CSV/TXT file from data/raw/."""
    encodings = ["utf-8", "latin-1", "cp1252"]
    lines = None

    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue

    if not lines or len(lines) < 2:
        print(f"  [WARN] Skipping {filepath.name}: empty or unreadable")
        return []

    sep = detect_separator(lines[0])

    if sep == "FIXED":
        return parse_fixed_width(lines)

    # CSV / semicolon-separated
    rows = []
    header = [h.strip().upper() for h in lines[0].split(sep)]

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(sep)
        if len(parts) < len(header):
            continue

        row = {}
        for i, col in enumerate(header):
            norm = COLUMN_MAP.get(col)
            if norm:
                val = parts[i].strip() if i < len(parts) else ""
                # Clean numeric values
                if norm not in ("seance", "groupe", "code_isin", "symbol"):
                    val = val.replace(",", ".").replace(" ", "")
                row[norm] = val

        if "seance" in row and "cloture" in row:
            rows.append(row)

    return rows


def parse_date(date_str: str) -> str | None:
    """Parse date from various BVMT formats."""
    date_str = date_str.strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def safe_float(val: str) -> float | None:
    """Convert string to float, return None on failure."""
    try:
        v = float(val.replace(",", ".").replace(" ", ""))
        return v if v >= 0 else None
    except (ValueError, AttributeError):
        return None


def safe_int(val: str) -> int:
    """Convert string to int, return 0 on failure."""
    try:
        return int(float(val.replace(",", ".").replace(" ", "")))
    except (ValueError, AttributeError):
        return 0


def main():
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed. Run:")
        print("  pip install psycopg2-binary")
        sys.exit(1)

    # Connect
    print(f"Connecting to PostgreSQL at {DB_HOST}:{DB_PORT}/{DB_NAME} as {DB_USER}...")
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
        )
        conn.autocommit = False
        cur = conn.cursor()
        print("[OK] Connected!\n")
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        print("\nMake sure PostgreSQL is running and the 'fixtrade' database exists.")
        print("To create it, run:  psql -U postgres -c \"CREATE DATABASE fixtrade;\"")
        sys.exit(1)

    # Find all data files
    files = sorted(RAW_DIR.glob("*.*"))
    if not files:
        print(f"No files found in {RAW_DIR}")
        sys.exit(1)

    print(f"Found {len(files)} files in {RAW_DIR}\n")

    total_inserted = 0
    total_skipped = 0

    insert_sql = """
        INSERT INTO stock_prices
            (symbol, code_isin, groupe, seance, ouverture, cloture,
             plus_bas, plus_haut, quantite_negociee, nb_transaction, capitaux)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, seance) DO NOTHING
    """

    for filepath in files:
        print(f"[FILE] Processing {filepath.name}...")
        rows = parse_csv(filepath)
        file_inserted = 0
        file_skipped = 0

        for row in rows:
            seance = parse_date(row.get("seance", ""))
            cloture = safe_float(row.get("cloture", ""))

            if not seance or cloture is None or cloture <= 0:
                file_skipped += 1
                continue

            symbol = row.get("symbol", "").strip()
            if not symbol:
                file_skipped += 1
                continue

            try:
                cur.execute(insert_sql, (
                    symbol,
                    row.get("code_isin", ""),
                    row.get("groupe", ""),
                    seance,
                    safe_float(row.get("ouverture", "")),
                    cloture,
                    safe_float(row.get("plus_bas", "")),
                    safe_float(row.get("plus_haut", "")),
                    safe_int(row.get("quantite_negociee", "0")),
                    safe_int(row.get("nb_transaction", "0")),
                    safe_float(row.get("capitaux", "0")),
                ))
                file_inserted += 1
            except Exception as e:
                print(f"  [WARN] Row error: {e}")
                conn.rollback()
                file_skipped += 1
                continue

        conn.commit()
        total_inserted += file_inserted
        total_skipped += file_skipped
        print(f"   [OK] Inserted: {file_inserted}, Skipped: {file_skipped}")

    # Summary
    print(f"\n{'='*50}")
    print(f"TOTAL INSERTED: {total_inserted:,}")
    print(f"TOTAL SKIPPED:  {total_skipped:,}")

    # Quick verification
    cur.execute("SELECT COUNT(*) FROM stock_prices")
    count = cur.fetchone()[0]
    print(f"\nRows in stock_prices: {count:,}")

    cur.execute("SELECT COUNT(DISTINCT symbol) FROM stock_prices")
    symbols = cur.fetchone()[0]
    print(f"Distinct symbols:     {symbols}")

    cur.execute("SELECT MIN(seance), MAX(seance) FROM stock_prices")
    min_date, max_date = cur.fetchone()
    print(f"Date range:           {min_date} to {max_date}")

    cur.execute("""
        SELECT symbol, COUNT(*) as days, MIN(seance), MAX(seance)
        FROM stock_prices
        GROUP BY symbol
        ORDER BY days DESC
        LIMIT 10
    """)
    print("\nTop 10 stocks by data points:")
    print(f"{'Symbol':<20} {'Days':>6} {'From':>12} {'To':>12}")
    print("-" * 54)
    for row in cur.fetchall():
        print(f"{row[0]:<20} {row[1]:>6} {row[2]!s:>12} {row[3]!s:>12}")

    cur.close()
    conn.close()
    print("\n[OK] Done!")


if __name__ == "__main__":
    main()
