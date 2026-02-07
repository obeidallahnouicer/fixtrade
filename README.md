# FixTrade — Minimal FastAPI Template (WSGI)

A small, clean, modular FastAPI template with a WSGI entrypoint.

Quick start

- Install dependencies:

```bash
pip install -r requirements.txt
```

- Run locally (ASGI / dev):

```bash
python -m uvicorn app.main:app --reload
```

- Run as WSGI (production example with gunicorn):

```bash
pip install gunicorn
gunicorn app.wsgi:application -w 4
```

- Or with waitress:

```bash
pip install waitress
waitress-serve --port=8080 app.wsgi:application
```

Run tests:

```bash
pytest -q
```
