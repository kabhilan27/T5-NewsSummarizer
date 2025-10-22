FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# If your app loads a local model folder (e.g., exported_t5_news), no change needed.
# If you use a Hub model id via env var MODEL_ID, you'll set it in Space Settings later.

EXPOSE 7860

# If your Flask entry is app.py and the instance is named `app`, this is correct:
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "300", "app:app"]
