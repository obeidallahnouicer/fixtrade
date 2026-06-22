#!/usr/bin/env python3
"""Simple smoke tester for key services. Writes JSON results to stdout or file."""
import json
import sys
import urllib.request
import urllib.error
from time import time

ENDPOINTS = {
    "api": "http://127.0.0.1:8000/api/v1/health",
    "ml": "http://127.0.0.1:8001/api/v1/health",
    "auth": "http://127.0.0.1:8002/api/v1/health",
    "genai": "http://127.0.0.1:8003/api/v1/health",
}


def check(url, timeout=3):
    out = {"url": url, "ok": False, "status": None, "body": None, "error": None}
    try:
        r = urllib.request.urlopen(url, timeout=timeout)
        out["status"] = r.getcode()
        try:
            body = r.read().decode("utf-8")
            out["body"] = json.loads(body)
        except Exception:
            out["body"] = body
        out["ok"] = 200 <= out["status"] < 400
    except urllib.error.HTTPError as e:
        out["status"] = e.code
        out["error"] = str(e)
    except Exception as e:
        out["error"] = str(e)
    return out


def main():
    results = {"timestamp": int(time()), "checks": {}}
    for k, u in ENDPOINTS.items():
        results["checks"][k] = check(u)

    if len(sys.argv) > 1:
        out_path = sys.argv[1]
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(out_path)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
