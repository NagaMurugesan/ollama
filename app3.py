import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama

# Load persisted FAISS index
storage_context = StorageContext.from_defaults(persist_dir="./embeddings")
index = load_index_from_storage(storage_context)

# Local model via Ollama (make sure you have ollama + llama3 pulled)
llm = Ollama(model="llama3")
query_engine = index.as_query_engine(llm=llm)

# Guardrail keywords
ALLOWED_KEYWORDS = [
    "pistachio", "nut", "orchard", "tree", "soil", "irrigation",
    "fertilizer", "harvest", "planting", "pruning", "rootstock"
]

def is_relevant(query: str) -> bool:
    query_lower = query.lower()
    return any(word in query_lower for word in ALLOWED_KEYWORDS)

# Streamlit UI
st.set_page_config(page_title="Pistachio Growing Assistant", layout="centered")
st.title("🌳 Pistachio Growing Assistant")
st.write("Ask me anything about growing pistachios!")

# Wrap in form so Enter works
with st.form(key="query_form"):
    query = st.text_input("Your question:", key="input")
    submit_button = st.form_submit_button("Ask")

if submit_button and query:
    if not is_relevant(query):
        st.warning("⚠️ Sorry, I can only answer questions about pistachio growing.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = query_engine.query(query)
                st.success(response.response)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
