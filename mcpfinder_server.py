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
import traceback

# --- huggingface assets ---
ASSETS = {
    "mcp_index.faiss": "https://huggingface.co/sohomx/mcpfinder-assets/resolve/main/mcp_index.faiss",
    "mcp_metadata.pkl": "https://huggingface.co/sohomx/mcpfinder-assets/resolve/main/mcp_metadata.pkl"
}
# --- download remote file to temp path ---
def download_temp(url):
    print(f"⬇️  downloading from {url}")
    r = requests.get(url)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(r.content)
    tmp.close()
    return tmp.name

# --- environment setup ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
assert openai.api_key, "❌ OPENAI_API_KEY is missing"

# --- load remote index + metadata ---
try:
    index_path = download_temp(ASSETS["mcp_index.faiss"])
    metadata_path = download_temp(ASSETS["mcp_metadata.pkl"])
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        mcps = pickle.load(f)
    print("✅ FAISS index + metadata loaded")
except Exception as e:
    print("❌ failed to load index or metadata")
    traceback.print_exc()
    raise e

# --- embedding function ---
def embed_query(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding, dtype="float32")

# --- semantic search ---
def search_mcp(task, top_k=5):
    vec = embed_query(task).reshape(1, -1)
    D, I = index.search(vec, top_k)
    return [mcps[i] for i in I[0]]

# --- fastapi setup ---
app = FastAPI()
print("🚀 starting mcpfinder_server.py...")

class MCPQuery(BaseModel):
    input: dict

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