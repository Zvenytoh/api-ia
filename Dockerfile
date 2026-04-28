FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_v2.py .
COPY modele_sentiment.pkl .

ENV PORT=5000
EXPOSE 5000

CMD gunicorn app_v2:app --bind 0.0.0.0:${PORT} --workers 2
