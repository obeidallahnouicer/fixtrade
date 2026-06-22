#!/usr/bin/env python3
"""Simple scraper worker wrapper.

Runs `scrapy crawl millim` in a loop and updates `data/last_scrape.txt` with
the UNIX timestamp of the last successful run. Sleeps between runs.
"""
import subprocess
import time
import os
import sys


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
LAST_SCRAPE = os.path.join(DATA_PATH, "last_scrape.txt")
SLEEP_SECONDS = int(os.environ.get("SCRAPER_INTERVAL_SECONDS", 3600))


def write_timestamp():
    os.makedirs(DATA_PATH, exist_ok=True)
    with open(LAST_SCRAPE, "w") as f:
        f.write(str(int(time.time())))


def run_crawl():
    try:
        print("Starting scrapy crawl...")
        res = subprocess.run(["scrapy", "crawl", "millim"], check=False)
        if res.returncode == 0:
            print("Crawl succeeded, updating timestamp")
            write_timestamp()
        else:
            print(f"Crawl failed with return code {res.returncode}")
    except FileNotFoundError:
        print("scrapy executable not found in container", file=sys.stderr)


def main():
    print("Scraper worker started")
    try:
        while True:
            run_crawl()
            time.sleep(SLEEP_SECONDS)
    except KeyboardInterrupt:
        print("Scraper worker stopping")


if __name__ == "__main__":
    main()
