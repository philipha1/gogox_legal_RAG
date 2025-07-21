import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import time
from openai import OpenAI
import os

# Streamlit page configuration
st.set_page_config(page_title="GoGoX RAG Q&A", page_icon="🤖")
st.title("GoGoX Regulatory and Disclosure Q&A App")
st.write("Ask questions based on HKEX Main Board Listing Rules and GoGoX disclosure documents.")

# Function to rewrite query for better retrieval
def rewrite_query(query: str) -> str:
    query = query.lower()
    if "size test" in query:
        query += " under Rule 14.07 of the HKEX Main Board Listing Rules"
    elif "public float" in query:
        query += " under Rule 8.08 of the HKEX Main Board Listing Rules"
    elif "disclosure" in query:
        query += " related to GoGoX announcements"
    return query

# OpenAI API key setup with debugging
st.write("Secrets loaded:", st.secrets)  # For debugging
st.write("Current directory:", os.path.dirname(__file__))  # Path debugging
OPENAI_API_KEY = st.secrets.get("secrets", {}).get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")  # Support environment variable for local hosting
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not configured. Please contact the administrator.")
    st.stop()
# Debugging: Verify OpenAI client initialization
st.write("Creating OpenAI client with API key:", OPENAI_API_KEY)
try:
    client = OpenAI(api_key=OPENAI_API_KEY)  # No http_client to avoid proxies error
    st.write("OpenAI client initialized successfully")
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
    
    # 3. Load embedding model
    try:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lighter model
        st.write("Embedding model loaded successfully")  # Debugging
    except Exception as e:
        st.error(f"Embedding model loading error: {e}")
        return None, None, None
    
    # 4. Prepare corpus with structured chunking
    max_text_length = 800  # Increased for richer context
    corpus = [
        f"[{r.get('rule_id', 'N/A')}] {r.get('title', 'No title')}:\n{r.get('text', 'No text')[:max_text_length]}"
        for r in deduplicated_rules
    ]
    st.write(f"Prepared corpus with {len(corpus)} entries")  # Debugging
    
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
    
    # 6. Create FAISS index
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, 32)  # Use HNSW for memory efficiency
        index.add(embeddings)
        st.write("FAISS index created successfully")  # Debugging
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
    # 1. Rewrite query for better retrieval
    rewritten_query = rewrite_query(query)
    st.write(f"Rewritten query: {rewritten_query}")  # Debugging
    
    # 2. Encode query and search FAISS
    query_vec = embedding_model.encode([rewritten_query], device="cpu")
    D, I = index.search(query_vec, k=top_k)
    
    # 3. Create structured context from top results
    retrieved_context = "\n\n".join([
        f"Context Document {i+1} - [{deduplicated_rules[i]['rule_id']}] {deduplicated_rules[i].get('title', 'N/A')}\n-----\n{deduplicated_rules[i]['text'][:800]}"
        for i in I[0]
    ])
    
    # 4. Enhanced OpenAI prompt
    prompt = f"""
You are a senior legal and compliance expert at GoGoX, specialized in HKEX Main Board Listing Rules and regulatory filings. Your job is to provide clear, concise, and accurate answers based strictly on the provided context from GoGoX's official disclosures and HKEX rules.

Follow these instructions:
1. Explain the relevant rules or disclosures step-by-step, citing specific clause numbers (e.g., Rule 14.07(1)) or announcement titles when available.
2. If the question involves calculations (e.g., size tests), show the formula, explain each step, and provide the result.
3. Use only the context below. If the answer is not found, state clearly: "The provided context does not contain sufficient information to answer the question."
4. Structure your answer professionally, with bullet points or numbered steps for clarity.

GoGoX Official Disclosure Extract:
-----------------------------------
{retrieved_context}

Question:
{query}

Answer:
"""
    
    # 5. Call OpenAI API
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a senior legal assistant specialized in HKEX regulations."},
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
