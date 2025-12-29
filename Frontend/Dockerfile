FROM python:3.11-slim AS frontend
WORKDIR /frontend
RUN apt-get update && apt-get install -y libgl1 dos2unix && rm -rf /var/lib/apt/lists/*
COPY Frontend ./Frontend
COPY Common ./Common
RUN find . -type f -name "*.py" -exec dos2unix {} + || true
RUN pip install --no-cache-dir -r Frontend/requirements.txt -r Common/requirements.txt
EXPOSE 8501
ENV BACKEND_URL=http://backend:8000 \
    API_SECRET_KEY=SuperSecureSecretKey123
CMD ["streamlit", "run", "Frontend/app.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
