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
        index = faiss.IndexHNSWFlat(dimension, 32)  # Use HNSW for memory efficiency
        index.add(embeddings)
        st.write(f"Memory usage after creating FAISS index: {get_memory_usage():.2f} MB")  # Debugging
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
    
    # 3. OpenAI prompt (original prompt preserved)
    prompt = f"""
You are a legal and compliance expert at GoGoX, a listed company on the Hong Kong Stock Exchange.
You are responsible for answering internal and external queries based strictly on GoGoX’s official disclosures submitted to the Stock Exchange.

Use only the information provided below, which comes from GoGoX's official filings and announcements on the HKEX website.
Do not speculate, do not guess, and do not use any external knowledge. Stick only to what is available in the context.

When appropriate, cite the relevant section or announcement title that supports your answer.

GoGoX Official Disclosure Extract:
-----------------------------------
{retrieved_context}

Question:
{query}

Answer:
"""
    
    # 4. Call OpenAI API
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a legal assistant."},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content, [deduplicated_rules[i] for i in I[0]]
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None, []

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input for questions
if user_query := st.chat_input("Enter your question (e.g., What is the minimum public float percentage required by HKEX?)"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    with st.spinner("Generating answer..."):
        answer, retrieved_docs = ask_openai_once(
            query=user_query,
            embedding_model=embedding_model,
            index=index,
            deduplicated_rules=deduplicated_rules
        )
        
        if answer:
            # Display answer
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Display retrieved documents
            with st.expander("View Retrieved Documents"):
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Document {i+1}** (ID: {doc['rule_id']}, Source: {doc['source']}): {doc.get('title', 'N/A')}")
                    st.markdown(f"{doc['text'][:200]}...")
        else:
            st.error("Failed to generate answer. Please try again.")

# Sidebar: App information
with st.sidebar:
    st.header("GoGoX RAG App")
    st.write("A question-answering system based on HKEX disclosures and GoGoX documents.")
    st.write("Contact: [Company email or contact person]")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and LangChain. GoGoX dedicated RAG app.")
