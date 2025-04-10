FROM python:3.11-slim

WORKDIR /app
COPY . /app

# install git-lfs and pull LFS-tracked files (like .faiss, .pkl)
RUN apt-get update && apt-get install -y git-lfs && \
    git lfs install && \
    git lfs pull

RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "mcpfinder_server:app", "--bind", "0.0.0.0:8000"]