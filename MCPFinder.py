import openai
import faiss
import pickle
import numpy as np
import os
import argparse
import json
from dotenv import load_dotenv

# Load OpenAI key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
assert openai.api_key, "❌ OPENAI_API_KEY missing from .env"

# Load FAISS index + metadata
index = faiss.read_index("mcp_index.faiss")
with open("mcp_metadata.pkl", "rb") as f:
    mcps = pickle.load(f)

# Embed query
def embed_query(query, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=query, model=model)
    return np.array(response.data[0].embedding, dtype="float32")

# Semantic search
def search_mcp(query, top_k=5):
    q_vec = embed_query(query).reshape(1, -1)
    D, I = index.search(q_vec, top_k)
    return [mcps[i] for i in I[0]]

# Simulate "mcp run ..." command
def simulate_mcp_run(mcp_url, input_dict):
    input_str = json.dumps(input_dict)
    print(f"\n⚙️  Simulated MCP Command:")
    print(f"mcp run {mcp_url} --input '{input_str}'")

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Natural language task")
    args = parser.parse_args()

    print(f"\n🔍 Searching for MCPs matching: \"{args.query}\"\n")
    matches = search_mcp(args.query)

    for i, mcp in enumerate(matches):
        print(f"#{i+1} — {mcp['name']}")
        print(f"   🔗 {mcp['url']}")
        print(f"   🧠 {mcp['description']}\n")

    # Simulate running the top match
    top = matches[0]
    simulate_mcp_run(top["url"], {"code": "1 + 2"})  # customize input_dict if needed