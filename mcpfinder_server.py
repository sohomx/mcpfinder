from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import faiss
import pickle
import numpy as np
import os
import json
import requests

# download index + metadata if missing
ASSETS = {
    "mcp_index.faiss": "https://huggingface.co/sohomx/mcpfinder-assets/resolve/main/mcp_index.faiss",
    "mcp_metadata.pkl": "https://huggingface.co/sohomx/mcpfinder-assets/resolve/main/mcp_metadata.pkl"
}

for filename, url in ASSETS.items():
    if not os.path.exists(filename):
        print(f"⬇️  downloading {filename} from HF...")
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)

# Load keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load index and metadata
index = faiss.read_index("mcp_index.faiss")
with open("mcp_metadata.pkl", "rb") as f:
    mcps = pickle.load(f)

# FastAPI setup
app = FastAPI()

# Input schema for /run
class MCPQuery(BaseModel):
    input: dict

# Embed query
def embed_query(query, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=query, model=model)
    return np.array(response.data[0].embedding, dtype="float32")

# Search logic
def search_mcp(query, top_k=5):
    q_vec = embed_query(query).reshape(1, -1)
    D, I = index.search(q_vec, top_k)
    return [mcps[i] for i in I[0]]

# Metadata endpoint
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

# Run endpoint
@app.post("/run")
def run(query: MCPQuery):
    task = query.input.get("task")
    if not task:
        return {"error": "Missing 'task' in input."}

    results = search_mcp(task)
    return {
        "results": [
            {
                "name": r["name"],
                "url": r["url"],
                "description": r["description"]
            }
            for r in results
        ]
    }

@app.post("/tool_call")
def tool_call(query: MCPQuery):
    return run(query)