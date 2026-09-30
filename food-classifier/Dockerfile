FROM python:3.11-slim

WORKDIR /app

COPY food-classifier/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY food-classifier/ /app/food-classifier/
COPY data/metadata/ /app/data/metadata/
WORKDIR /app/food-classifier

EXPOSE 7860
CMD ["python", "src/app.py"]
