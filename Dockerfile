# One image, two processes (pick with the start command):
#   API + worker:  uvicorn verdict.api:app --host 0.0.0.0 --port ${PORT:-8000}
#   UI:            streamlit run ui/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY verdict ./verdict
RUN pip install --no-cache-dir ".[ui]"
COPY . .
EXPOSE 8000
CMD ["sh", "-c", "uvicorn verdict.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
