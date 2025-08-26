import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama

# Load RAG index
storage_context = StorageContext.from_defaults(persist_dir="./embeddings")
index = load_index_from_storage(storage_context)

llm = Ollama(model="llama3")
query_engine = index.as_query_engine(llm=llm)

# --- Guardrails ---
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedder = SentenceTransformer("all-MiniLM-L6-v2")
REFERENCE_SENTENCES = [
    "How to grow pistachios",
    "Best soil for pistachio trees",
    "Pistachio irrigation methods",
    "Pistachio pruning and harvesting",
    "Pistachio nut diseases and pests",
    "Fertilizer for pistachio orchards",
    "How to plant pistachio trees"
]
reference_embeddings = embedder.encode(REFERENCE_SENTENCES)

ALLOWED_KEYWORDS = [
    "pistachio", "nut", "orchard", "tree", "soil",
    "irrigation", "fertilizer", "harvest", "planting",
    "pruning", "rootstock", "pests", "disease"
]

def is_keyword_relevant(query: str) -> bool:
    return any(word in query.lower() for word in ALLOWED_KEYWORDS)

def is_semantically_relevant(query: str, threshold: float = 0.55) -> bool:
    query_emb = embedder.encode([query])
    sims = cosine_similarity(query_emb, reference_embeddings)
    return np.max(sims) >= threshold

def is_relevant(query: str) -> bool:
    return is_keyword_relevant(query) or is_semantically_relevant(query)

# --- Streamlit UI ---
st.set_page_config(page_title="Pistachio Growing Assistant", layout="centered")
st.title("🌳 Pistachio Growing Assistant")
st.write("Ask me anything about growing pistachios!")

query = st.text_input("Your question:")

if st.button("Ask"):
    if not is_relevant(query):
        st.warning("⚠️ Sorry, I can only answer questions about pistachio growing.")
    else:
        response = query_engine.query(query)
        st.success(response.response)
