"""
WSGI compatibility layer.

Wraps the ASGI FastAPI application for deployment on WSGI servers
such as Gunicorn or Waitress. Prefer ASGI deployment when possible.
"""

from asgiref.wsgi import AsgiToWsgi

from app.main import app

# Expose a WSGI-compatible app object for WSGI servers (gunicorn, waitress, etc.)
application = AsgiToWsgi(app)
