FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements-scraper.txt ./
RUN pip install --no-cache-dir -r requirements-scraper.txt

COPY . /app

RUN useradd -m worker && chown -R worker:worker /app
USER worker

CMD ["python", "run_training.py"]
