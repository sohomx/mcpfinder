# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download FAISS + metadata index directly
RUN apt-get update && apt-get install -y curl && \
    curl -L -o mcp_index.faiss https://huggingface.co/sohomx/mcpfinder-assets/resolve/main/mcp_index.faiss && \
    curl -L -o mcp_metadata.pkl https://huggingface.co/sohomx/mcpfinder-assets/resolve/main/mcp_metadata.pkl

# Copy app files
COPY . .

# Expose and run
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "mcpfinder_server:app"]