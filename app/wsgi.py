from asgiref.wsgi import AsgiToWsgi
from app.main import app

# Expose a WSGI-compatible app object for WSGI servers (gunicorn, waitress, etc.)
application = AsgiToWsgi(app)
