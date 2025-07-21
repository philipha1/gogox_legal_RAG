import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import time
from openai import OpenAI
import os
import httpx
import traceback

# Streamlit page configuration
st.set_page_config(page_title="GoGoX RAG Q&A", page_icon="🤖")
st.title("GoGoX Regulatory and Disclosure Q&A App")
st.write("Ask questions based on HKEX Main Board Listing Rules and GoGoX disclosure documents.")

# OpenAI API key setup with debugging
st.write("Secrets file path:", os.path.join(os.path.dirname(__file__), ".streamlit/secrets.toml"))
try:
    st.write("Secrets loaded:", st.secrets)
except Exception as e:
    st.error(f"Secrets loading error: {e}")
    st.stop()
# Cloud와 로컬 모두 지원하는 Secrets 접근
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("secrets", {}).get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not configured. Please contact the administrator.")
    st.stop()
st.write("Creating OpenAI client with API key:", OPENAI_API_KEY)
try:
    http_client = httpx.Client(proxies=None)
    client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
    st.write("OpenAI client initialized successfully")
except Exception as e:
    st.error(f"OpenAI API error: {e}")
    st.write("Traceback:", traceback.format_exc())
    st.stop()

# JSON loading function with dynamic path adjustment
def load_json(file_path):
    base_path = os.path.dirname(__file__)
    full_path = os.path.join(base_path, file_path)
    st.write(f"Attempting to load: {full_path}")
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {full_path}")
        return []
    except json.JSONDecodeError:
        st.error(f"JSON parsing error: {full_path}")
        return []

# Load RAG pipeline with caching
@st.cache_resource
def load_rag_pipeline():
    start = time.time()
    
    # 1. Load JSON data with adjusted paths
    rules1 = load_json("all_rules_merged.json")
    rules2 = load_json("gogox_announcements.json")
    if not rules1 and not rules2:
        st.error("Unable to load JSON data.")
        return None, None, None
    
    # 2. Merge JSON and remove duplicates
    combined = rules1 + rules2
    unique_rules = {}
    for i, r in enumerate(combined):
        rid = r.get("rule_id", f"rules1_auto_{i}" if r in rules1 else f"rules2_auto_{i}")
        if rid in unique_rules:
            unique_rules[rid]["text"] = unique_rules[rid].get("text", "") + " | " + r.get("text", "")
            unique_rules[rid]["title"] = unique_rules[rid].get("title", "") or r.get("title", "")
        else:
            unique_rules[rid] = r.copy()
            unique_rules[rid]["rule_id"] = rid
            unique_rules[rid]["source"] = "rules1" if r in rules1 else "rules2"
    deduplicated_rules = list(unique_rules.values())
    
    # 3. Load embedding model
    try:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        st.write("Embedding model loaded successfully")
    except Exception as e:
        st.error(f"Embedding model loading error: {e}")
        return None, None, None
    
    # 4. Prepare corpus
    max_text_length = 300
    corpus = [
        f"{r.get('title', r.get('text', '')[:30] or 'No title')}. {r.get('text', '')[:max_text_length] or 'No text'}"
        for r in deduplicated_rules
    ]
    
    # 5. Generate embeddings
    batch_size = 16
    embeddings = []
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        batch_embeddings = embedding_model.encode(
            batch, batch_size=batch_size, device="cpu"
        )
        embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)
    
    # 6. Create FAISS index
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        st.write("FAISS index created successfully")
    except Exception as e:
        st.error(f"FAISS index creation error: {e}")
        return None, None, None
    
    st.write(f"✅ FAISS index loaded successfully: {len(corpus)} rules (Time: {time.time() - start:.2f} seconds)")
    return embedding_model, index, deduplicated_rules

# Load RAG pipeline
embedding_model, index, deduplicated_rules = load_rag_pipeline()

# Stop app if pipeline loading fails
if embedding_model is None or index is None or deduplicated_rules is None:
    st.error("RAG pipeline initialization failed. Please contact the administrator.")
    st.stop()

# Question-answering function
def ask_openai_once(query: str, embedding_model, index, deduplicated_rules, top_k: int = 5, model="gpt-4"):
    # 1. Encode query and search FAISS
    query_vec = embedding_model.encode([query], device="cpu")
    D, I = index.search(query_vec, k=top_k)
    
    # 2. Create context from top results
    retrieved_context = "\n\n".join([
        f"[{deduplicated_rules[i]['rule_id']}] {deduplicated_rules[i].get('title', 'N/A')}\n{deduplicated_rules[i]['text']}"
        for i in I[0]
    ])
    
    # 3. OpenAI prompt
    prompt = f"""
