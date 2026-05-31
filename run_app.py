#!/usr/bin/env python3
"""
App launcher that ensures .env is loaded before starting the app.

This solves the issue where DATABASE_URL environment variable is not set
when uvicorn starts the app directly.
"""

import os
import sys
from pathlib import Path

# Load .env file explicitly BEFORE importing the app
from dotenv import load_dotenv

# Get the project root directory
project_root = Path(__file__).parent.absolute()
env_file = project_root / ".env"

if env_file.exists():
    print(f"✓ Loading environment from {env_file}")
    load_dotenv(env_file, override=True)
else:
    print(f"⚠ Warning: .env file not found at {env_file}")
    print("  Proceeding with system environment variables only")

# Verify DATABASE_URL is set
db_url = os.getenv("DATABASE_URL")
if not db_url:
    print("✗ ERROR: DATABASE_URL not set after loading .env")
    print(f"  Checked: {env_file}")
    sys.exit(1)

print(f"✓ DATABASE_URL is set: {db_url[:50]}...")

# Now import and run the app
if __name__ == "__main__":
    import uvicorn
    
    # Run with reload enabled for development
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
