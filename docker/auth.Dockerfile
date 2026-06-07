FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements-auth.txt ./
RUN pip install --no-cache-dir -r requirements-auth.txt

COPY . /app

RUN useradd -m authuser && chown -R authuser:authuser /app
USER authuser

EXPOSE 8002

CMD ["uvicorn", "services.auth_service.app:app", "--host", "0.0.0.0", "--port", "8002"]
