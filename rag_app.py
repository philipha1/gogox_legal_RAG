import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import time
from openai import OpenAI
import os
import psutil  # For memory usage debugging

# Streamlit page configuration
st.set_page_config(page_title="GoGoX RAG Q&A", page_icon="🤖")
st.title("GoGoX Regulatory and Disclosure Q&A App")
st.write("Ask questions based on HKEX Main Board Listing Rules and GoGoX disclosure documents.")

# Function to get current memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Return memory in MB

# OpenAI API key setup with debugging
st.write("Secrets loaded:", st.secrets)  # For debugging
st.write("Current directory:", os.path.dirname(__file__))  # Path debugging
st.write(f"Memory usage before OpenAI init: {get_memory_usage():.2f} MB")  # Debugging
OPENAI_API_KEY = st.secrets.get("secrets", {}).get("OPENAI_API_KEY")  # Extract from nested secrets
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not configured. Please contact the administrator.")
    st.stop()
# Debugging: Verify OpenAI client initialization
st.write("Creating OpenAI client with API key:", OPENAI_API_KEY)
try:
    client = OpenAI(api_key=OPENAI_API_KEY)  # No http_client to avoid proxies error
    st.write("OpenAI client initialized successfully")
    st.write(f"Memory usage after OpenAI init: {get_memory_usage():.2f} MB")  # Debugging
except Exception as e:
    st.error(f"OpenAI API error: {e}")
    st.stop()

# Function to load JSON files with dynamic path adjustment
def load_json(file_path):
    base_path = os.path.dirname(__file__)
    full_path = os.path.join(base_path, file_path)
    st.write(f"Attempting to load: {full_path}")  # Debugging
    if not os.path.exists(full_path):
        st.error(f"JSON file does not exist: {full_path}")
        return []
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            st.write(f"Successfully loaded {file_path} with {len(data)} entries")  # Debugging
            # Limit to 500 entries to reduce memory usage
            data = data[:500]
            return data
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
    st.write(f"Memory usage before loading JSON: {get_memory_usage():.2f} MB")  # Debugging
    
    # 1. Load JSON data
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
    st.write(f"Memory usage after merging JSON: {get_memory_usage():.2f} MB")  # Debugging
    
    # 3. Load embedding model
    try:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lighter model
        st.write(f"Memory usage after loading model: {get_memory_usage():.2f} MB")  # Debugging
    except Exception as e:
        st.error(f"Embedding model loading error: {e}")
        return None, None, None
    
    # 4. Prepare corpus
    max_text_length = 100  # Reduced for memory optimization
    corpus = [
        f"{r.get('title', r.get('text', '')[:30] or 'No title')}. {r.get('text', '')[:max_text_length] or 'No text'}"
        for r in deduplicated_rules
    ]
    st.write(f"Prepared corpus with {len(corpus)} entries")  # Debugging
    st.write(f"Memory usage after preparing corpus: {get_memory_usage():.2f} MB")  # Debugging
    
    # 5. Generate embeddings
    batch_size = 8  # Reduced for memory optimization
    embeddings = []
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        batch_embeddings = embedding_model.encode(
            batch, batch_size=batch_size, device="cpu"
        )
        embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)
    st.write(f"Memory usage after generating embeddings: {get_memory_usage():.2f} MB")  # Debugging
    
    # 6. Create FAISS index
    try:
        dimension = embeddings.shape[1]
        index = faiss.Index
