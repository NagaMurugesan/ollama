import streamlit as st
from streamlit_chat import message
#from src.validate import validate_openai_key
#from src.chat_interface import text_based
#from src.model import csv_agent
import pandas as pd
import openai
import os

import os
import io
import time
from typing import List
from dotenv import load_dotenv
import nltk
import pdfplumber

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

import html
# Download stopwords (only needs to be done once)
# Ensure required nltk resources exist
for resource in ["stopwords", "punkt", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download("wordnet")


import streamlit as st

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.schema import TextNode

PERSIST_DIR = "storage"
DATA_DIR = "data"

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)

def configure_llamaindex():
    load_dotenv()
    llm_model = os.getenv("LLM_MODEL", "llama3")
    embed_model_name = os.getenv("EMBED_MODEL", "nomic-embed-text")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
    chunk_size = int(os.getenv("CHUNK_SIZE", "1024"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "128"))

    # Configure global settings
    print(system_prompt)
    print(llm_model)
    print(embed_model_name)

    Settings.llm = Ollama(model=llm_model, base_url=base_url, request_timeout=120, system_prompt=system_prompt)
    Settings.embed_model = OllamaEmbedding(model_name=embed_model_name, base_url=base_url)
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

def load_or_build_index():
    if os.path.isdir(PERSIST_DIR) and any(os.scandir(PERSIST_DIR)):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_context)
    else:
        # First run: build from data dir if present
        docs = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()
        if not docs:
            # empty index (we'll allow uploads)
            return VectorStoreIndex([])
        index = VectorStoreIndex.from_documents(docs, show_progress=True)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        return index

def append_uploaded_files_to_index(index: VectorStoreIndex, uploaded_files) -> int:
    """Save uploaded files to data/, parse, and add to index incrementally."""
    if not uploaded_files:
        return 0

    saved_paths: List[str] = []
    for uf in uploaded_files:
        # Save to data/
        path = os.path.join(DATA_DIR, uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        saved_paths.append(path)

    # Read the new docs and upsert
    new_docs = SimpleDirectoryReader(input_files=saved_paths).load_data()
    if not new_docs:
        return 0

    # Convert to Nodes and insert
    nodes: List[TextNode] = []
    for d in new_docs:
        nodes.append(TextNode(text=d.get_content(), metadata=d.metadata))

    index.insert_nodes(nodes)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return len(new_docs)

def render_sources(response) -> None:
    # LlamaIndex response often has source_nodes
    if not hasattr(response, "source_nodes") or not response.source_nodes:
        return
    st.markdown("#### Sources")
    for i, sn in enumerate(response.source_nodes, start=1):
        meta = sn.node.metadata or {}
        fname = meta.get("file_name") or meta.get("filename") or meta.get("source") or "document"
        score = f"{sn.score:.3f}" if sn.score is not None else "-"
        with st.expander(f"{i}. {fname} (score {score})", expanded=False):
            st.write(sn.node.get_content()[:1200] + ("..." if len(sn.node.get_content()) > 1200 else ""))



# Streamlit app main function
def main_ui():

            # Query UI
    st.subheader("Ask a question")
    top_k = int(os.getenv("TOP_K", "4"))

    # Use form so hitting Enter submits
    with st.form(key="query_form"):
        q = st.text_input("e.g., What soil type and pH are best for pistachios?", key="user_query")
        submit_btn = st.form_submit_button("Ask")

    if submit_btn and q.strip():
        qe = st.session_state.index.as_query_engine(similarity_top_k=top_k)
        with st.spinner("Thinking locally..."):
            response = qe.query(q)

        st.markdown("### Answer")
        st.write(response.response if hasattr(response, "response") else str(response))
        render_sources(response)


    # col1, col2, col3, col4 = st.columns((1, 3, 3.5, 2.5))
    # c1, c2, c3 = st.columns((1, 6.5, 2.5))

    # with col2:
    #     st.write('## ')
    #     st.write("### ")
    #     st.markdown(
    #         "<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Enter OpenAI API Key</span></p>", 
    #         unsafe_allow_html=True
    #     )
    # with col3:
    #     st.write("### ")
    #     vAR_api_key = st.text_input(" ", type="password")

    # if vAR_api_key:
    #     with col4:
    #         message = validate_openai_key(vAR_api_key)
    #         if "Invalid" in message:
    #             st.write("### ")
    #             st.write("### ")
    #             st.warning(message)
    #         else:
    #             st.write("### ")
    #             st.write("### ")
    #             st.success(message)
    #     if message == "Valid API Key!":
    #         with col2:
    #             st.write('## ')
    #             st.write("### ")
    #             st.markdown(
    #                 "<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload KnowledgeBase</span></p>", 
    #                 unsafe_allow_html=True
    #             )
    #         with col3:
    #             st.write("### ")
    #             uploaded_file  = st.file_uploader("", type='csv', key="fileupload")
    #         if uploaded_file:
    #             vAR_DB=csv_agent(uploaded_file, vAR_api_key)
    #             text_based(vAR_api_key, vAR_DB)
           
        
