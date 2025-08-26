import streamlit as st
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SimpleNodeParser
#from llama_index.readers.file import SimpleDirectoryReader
import os

# ---------- LLM + Embeddings Setup ----------
# embed_model = OllamaEmbedding(model_name="nomic-embed-text")
# llm = Ollama(model="llama2")
# service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# def load_or_build_index():
#     if os.path.exists("./storage"):
#         storage_context = StorageContext.from_defaults(persist_dir="./storage")
#         index = load_index_from_storage(storage_context, service_context=service_context)
#     else:
#         docs = SimpleDirectoryReader("./data").load_data()
#         parser = SimpleNodeParser.from_defaults()
#         nodes = parser.get_nodes_from_documents(docs)

#         index = VectorStoreIndex(nodes, service_context=service_context)
#         index.storage_context.persist("./storage")

#     return index

# INDEX = load_or_build_index()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="🌿 Pistachio Knowledge Bot", page_icon="🌿", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #e0f7fa, #fce4ec);
        font-family: 'Helvetica Neue', sans-serif;
    }
    .main-header {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #4a148c;
        margin-bottom: 1rem;
    }
    .query-box {
        background: #ffffffdd;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .response-box {
        background: #f3e5f5;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        margin-top: 1rem;
        font-size: 1.1rem;
        line-height: 1.6;
        color: #311b92;
    }
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #555;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown("<div class='main-header'>🌿 Pistachio Knowledge Assistant 🌿</div>", unsafe_allow_html=True)

# Input area
with st.container():
    st.markdown("<div class='query-box'>", unsafe_allow_html=True)
    user_query = st.text_area("🔍 Ask me anything about Pistachios:", height=100, placeholder="e.g., What insects attack pistachios?")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("✨ Get Answer"):
        if user_query.strip():
            # qe = INDEX.as_query_engine()
            # response = qe.query(user_query)

            st.markdown("<div class='response-box'>", unsafe_allow_html=True)
            #st.write(response.response)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a query before submitting.")

# Footer
st.markdown("<div class='footer'>💡 Powered by LlamaIndex + Ollama | Designed for eye-soothing experience 🌸</div>", unsafe_allow_html=True)
