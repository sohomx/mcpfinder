import openai
import faiss
import pickle
import numpy as np
import time
import os
from dotenv import load_dotenv
from parse_mcp_list import get_awesome_mcp_markdown, parse_markdown_for_mcps

# Load OpenAI API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
assert openai.api_key, "❌ OPENAI_API_KEY not found in .env"

# Load and parse MCPs
markdown = get_awesome_mcp_markdown()
mcps = parse_markdown_for_mcps(markdown)
texts = [f"{m['name']} - {m['description']}" for m in mcps]

print(f"📝 Starting to embed {len(texts)} MCPs...\n")

# Embedding function with retry
def get_embeddings(texts, model="text-embedding-3-small"):
    vectors = []
    for i, text in enumerate(texts):
        print(f"→ [{i+1}/{len(texts)}] Embedding: {text[:60]}...")
        for attempt in range(3):
            try:
                response = openai.embeddings.create(input=text, model=model)
                vec = response.data[0].embedding
                vectors.append(vec)
                break
            except Exception as e:
                print(f"⚠️  Error (attempt {attempt+1}): {e}")
                time.sleep(1)
        else:
            print(f"❌ Failed to embed: {text[:60]}")
            vectors.append([0.0]*1536)
        time.sleep(0.3)
    return vectors

# Embed all MCPs
embeddings = get_embeddings(texts)

# Build FAISS index
dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32"))

# Save index + metadata
with open("mcp_metadata.pkl", "wb") as f:
    pickle.dump(mcps, f)
faiss.write_index(index, "mcp_index.faiss")

print(f"\n✅ Indexed {len(mcps)} MCPs into FAISS using OpenAI embeddings.")