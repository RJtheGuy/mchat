import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import chunk_text, load_all_files

def generate_index():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.to('cpu')

    chunks = []
    metadatas = []

    print("📚 Processing knowledge base...")
    for filepath, text in load_all_files("knowledge_base"):
        for chunk in chunk_text(text, max_tokens=250):
            embedding = model.encode(chunk, convert_to_numpy=True)
            chunks.append(embedding)
            metadatas.append({'text': chunk, 'source': filepath})

    print(f"🧠 Built {len(chunks)} embeddings")
    index = faiss.IndexFlatL2(384)
    index.add(np.vstack(chunks))

    os.makedirs('faiss_index', exist_ok=True)
    with open('faiss_index/index.pkl', 'wb') as f:
        pickle.dump((index, metadatas), f)

    print("✅ FAISS index saved to faiss_index/index.pkl")

if __name__ == "__main__":
    generate_index()
