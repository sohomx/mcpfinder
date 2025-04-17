FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "mcpfinder_server:app", "--host", "0.0.0.0", "--port", "8000"]
