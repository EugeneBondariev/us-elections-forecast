FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY src/ src/
COPY data/ data/
COPY app.py predict.py api.py ./

# Default: run streamlit
EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
