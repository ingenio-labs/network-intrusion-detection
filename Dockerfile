# Serving image for the API. Train first (python -m src.train) so models/model.joblib exists, then:
#   docker build -t flow-api . && docker run -p 8000:8000 flow-api
FROM python:3.12-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src/ src/
COPY docs/examples/ docs/examples/
COPY models/model.joblib models/model.joblib
RUN useradd --create-home api
USER api
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
