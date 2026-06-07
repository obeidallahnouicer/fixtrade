FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements-genai.txt ./
RUN pip install --no-cache-dir -r requirements-genai.txt

COPY . /app

RUN useradd -m gaiuser && chown -R gaiuser:gaiuser /app
USER gaiuser

EXPOSE 8003

CMD ["uvicorn", "services.genai_service.app:app", "--host", "0.0.0.0", "--port", "8003"]
