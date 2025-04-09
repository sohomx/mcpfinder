import openai
import faiss
import pickle
import numpy as np
import os
import argparse
from dotenv import load_dotenv

# Load .env with OpenAI key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
assert openai.api_key, "❌ OPENAI_API_KEY missing from .env"

# Load FAISS + metadata
index = faiss.read_index("mcp_index.faiss")
with open("mcp_metadata.pkl", "rb") as f:
    mcps = pickle.load(f)

# Embed query using OpenAI
def embed_query(query, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=query, model=model)
    return np.array(response.data[0].embedding, dtype="float32")

# Search FAISS
def search_mcp(query, top_k=5):
    q_vec = embed_query(query).reshape(1, -1)
    D, I = index.search(q_vec, top_k)
    results = [mcps[i] for i in I[0]]
    return results

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Search query for an MCP")
    args = parser.parse_args()

    print(f"\n🔍 Searching MCPs for: \"{args.query}\"\n")
    matches = search_mcp(args.query)

    for i, mcp in enumerate(matches):
        print(f"#{i+1} — {mcp['name']}")
        print(f"   🔗 {mcp['url']}")
        print(f"   🧠 {mcp['description']}\n")