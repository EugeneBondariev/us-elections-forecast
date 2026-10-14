FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project. data/ is intentionally NOT copied here — it's gitignored (large
# source files + generated artifacts) and is expected to be bind-mounted at
# runtime, as docker-compose.yml already does (./data:/app/data and ./models:/app/models).
# Empty dirs are created (matching src/paths.py's raw/external/interim/processed split)
# so paths exist even before the mount.
COPY src/ src/
COPY app.py api.py ./
RUN mkdir -p data/raw data/external data/interim data/processed models reports/figures

# Default: run streamlit
EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
