import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import chunk_text, load_all_files

def generate_index():
    # Use a smaller, faster embedding model
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    model.to('cpu')
    
    chunks = []
    metadatas = []
    
    print("📚 Processing knowledge base...")
    for filepath, text in load_all_files("knowledge_base"):
        for chunk in chunk_text(text, max_tokens=200):  # Smaller chunks for faster processing
            embedding = model.encode(chunk, convert_to_numpy=True)
            chunks.append(embedding)
            metadatas.append({'text': chunk, 'source': filepath})
    
    print(f"🧠 Built {len(chunks)} embeddings")
    
    # Create and populate the index
    dimension = len(chunks[0])
    index = faiss.IndexFlatL2(dimension)
    
    # Convert to numpy array for more efficient processing
    chunk_array = np.vstack(chunks).astype('float32')
    index.add(chunk_array)
    
    # Optionally create IVF index for faster search on larger datasets
    if len(chunks) > 1000:
        print("🔍 Creating IVF index for faster search...")
        nlist = min(int(len(chunks) / 10), 100)  # Rule of thumb for nlist
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(chunk_array)
        index.add(chunk_array)
    
    os.makedirs('faiss_index', exist_ok=True)
    with open('faiss_index/index.pkl', 'wb') as f:
        pickle.dump((index, metadatas), f)
    
    print("✅ FAISS index saved to faiss_index/index.pkl")

if __name__ == "__main__":
    generate_index()