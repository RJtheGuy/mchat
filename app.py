import streamlit as st
import pickle
import faiss
import numpy as np
import time
import os
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from functools import lru_cache

# ====================
# Configuration
# ====================
MODEL_DIR = "model"
MODEL_FILENAME = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # your .gguf file
FAISS_INDEX_PATH = "faiss_index/index.pkl"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# ====================
# Model Loading
# ====================
@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer(EMBEDDING_MODEL)
    model.to('cpu')
    model.encode("warmup")
    return model

@st.cache_resource
def load_llm():
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    return Llama(
        model_path=model_path,
        n_ctx=1024,
        n_threads=6,
        verbose=False
    )

@st.cache_resource
def load_faiss_index():
    with open(FAISS_INDEX_PATH, "rb") as f:
        return pickle.load(f)

# ====================
# Core Functions
# ====================
@lru_cache(maxsize=1000)
def cached_encode(model, text):
    return model.encode(text, convert_to_numpy=True)

def cached_retrieve(index, metadatas, query_vec):
    query_vec = np.array(query_vec, dtype='float32')
    _, indices = index.search(query_vec.reshape(1, -1), 2)
    return [metadatas[i]['text'] for i in indices[0]]

def generate_response(llm, query, context_chunks):
    context = "\n".join(f"- {chunk[:300]}" for chunk in context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    try:
        result = llm(prompt, max_tokens=100, temperature=0.2, stop=["\n"])
        return result["choices"][0]["text"].strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ====================
# App Initialization
# ====================
embedding_model = load_embedding_model()
llm = load_llm()
index, metadatas = load_faiss_index()

# ====================
# Streamlit UI
# ====================
st.title("🤖 Portfolio Assistant")
query = st.text_input("Ask about my projects or experience:")

if query:
    start_time = time.time()
    
    with st.spinner("🔍 Searching..."):
        query_vec = cached_encode(embedding_model, query).astype('float32')
        context = cached_retrieve(index, metadatas, query_vec)
        retrieval_time = time.time() - start_time

    gen_start = time.time()
    answer = generate_response(llm, query, context)
    gen_time = time.time() - gen_start

    response = st.empty()
    text = ""
    for word in answer.split():
        text += word + " "
        response.markdown(f"**Answer:** {text}")
        time.sleep(0.02)

    st.caption(f"⏱️ Retrieval: {retrieval_time:.2f}s | Generation: {gen_time:.2f}s | Total: {time.time()-start_time:.2f}s")
