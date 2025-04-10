from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import faiss
import pickle
import numpy as np
import os
import requests
import tempfile

# --- remote asset config ---
HF_REPO = "sohomx/mcpfinder-assets"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"

ASSETS = {
    "mcp_index.faiss": "https://huggingface.co/datasets/sohomx/mcpfinder-assets/resolve/main/mcp_index.faiss",
    "mcp_metadata.pkl": "https://huggingface.co/datasets/sohomx/mcpfinder-assets/resolve/main/mcp_metadata.pkl"
}

# --- utility: download to temp file ---
def download_temp(url):
    print(f"⬇️  downloading: {url}")
    r = requests.get(url)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(r.content)
        return f.name

# --- load index + metadata ---
def load_assets():
    index_path = download_temp(ASSETS["index"])
    meta_path = download_temp(ASSETS["meta"])

    idx = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        data = pickle.load(f)
    return idx, data

# --- embed text using OpenAI ---
def embed_query(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding, dtype="float32")

# --- search top K from faiss ---
def search_mcp(task, top_k=5):
    vec = embed_query(task).reshape(1, -1)
    D, I = index.search(vec, top_k)
    return [mcps[i] for i in I[0]]

# --- env setup ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
assert openai.api_key, "❌ OPENAI_API_KEY is missing"

# --- load index + metadata ---
index, mcps = load_assets()

# --- fastapi app ---
app = FastAPI()

# --- schema ---
class MCPQuery(BaseModel):
    input: dict

# --- endpoints ---
@app.get("/metadata")
def metadata():
    return {
        "name": "mcpfinder",
        "description": "Searches for the best MCP tool given a natural language task.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The user task in plain language"
                }
            },
            "required": ["task"]
        }
    }

@app.post("/run")
def run(query: MCPQuery):
    task = query.input.get("task")
    if not task:
        return {"error": "Missing 'task' in input."}
    results = search_mcp(task)
    return {
        "results": [
            {"name": r["name"], "url": r["url"], "description": r["description"]}
            for r in results
        ]
    }

@app.post("/tool_call")
def tool_call(query: MCPQuery):
    return run(query)