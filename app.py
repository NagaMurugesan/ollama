# app.py
import os
import io
import time
from typing import List
from dotenv import load_dotenv
import nltk
import pdfplumber

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

def main():
    st.set_page_config(page_title="Pistachio Assistant (Local RAG)", page_icon="🌰", layout="wide")
    st.title("🌰 Pistachio Growing Assistant (Local RAG)")
    st.caption("Runs 100% locally with Ollama + LlamaIndex. Drop your pistachio docs and ask questions.")

    ensure_dirs()
    configure_llamaindex()

    # Side bar config
    with st.sidebar:
        st.header("⚙️ Settings")
        st.write("Models (configured via .env):")
        st.code(f"LLM_MODEL={os.getenv('LLM_MODEL', 'llama3')}\nEMBED_MODEL={os.getenv('EMBED_MODEL', 'nomic-embed-text')}", language="env")
        st.write("Index Folder:")
        st.code(PERSIST_DIR)
        st.write("Data Folder:")
        st.code(DATA_DIR)

        st.divider()
        uploaded = st.file_uploader(
            "➕ Add more documents",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True
        )
        add_btn = st.button("Add to Index")

    # Load or build index
    if "index" not in st.session_state:
        with st.spinner("Loading index..."):
            st.session_state.index = load_or_build_index()

    # Handle uploads
    if uploaded and add_btn:
        with st.spinner("Indexing uploaded files..."):
            added = append_uploaded_files_to_index(st.session_state.index, uploaded)
        st.success(f"Indexed {added} new document(s).")

    # # Query UI
    # st.subheader("Ask a question")
    # q = st.text_input("e.g., What soil type and pH are best for pistachios?")
    # top_k = int(os.getenv("TOP_K", "4"))

    # if st.button("Ask") and q.strip():
    #     qe = st.session_state.index.as_query_engine(similarity_top_k=top_k)
    #     with st.spinner("Thinking locally..."):
    #         response = qe.query(q)

    #     st.markdown("### Answer")
    #     st.write(response.response if hasattr(response, "response") else str(response))
    #     render_sources(response)

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
        user_query=q

        if hasattr(response, "source_nodes") and response.source_nodes:
            st.markdown("#### Sources & Relevant Passages")
        query_terms = [term.lower() for term in user_query.split() if term.strip()]

        for i, sn in enumerate(response.source_nodes, start=1):
            meta = sn.node.metadata or {}
            fname = meta.get("file_name") or meta.get("filename") or meta.get("source") or "document"
            score = sn.score if sn.score is not None else 0
            color = "#a6cee3" if score > 0.8 else "#b2df8a" if score > 0.5 else "#fb9a99"

            content_preview = sn.node.get_content()
            lines = content_preview.split("\n")

            # Highlight query terms in snippet
            highlighted_lines = []
            for line in lines[:20]:
                hl_line = line
                for term in query_terms:
                    if term in line.lower():
                        hl_line = hl_line.replace(term, f"<span style='color:red;font-weight:bold'>{term}</span>")
                highlighted_lines.append(html.escape(hl_line))
            highlighted_text = "<br>".join(highlighted_lines)

            with st.expander(f"{i}. {fname} (score {score:.3f})", expanded=False):
                # Snippet preview
                st.markdown(f"<div style='background-color:{color};padding:5px;border-radius:5px'>{highlighted_text}</div>", unsafe_allow_html=True)

                # ------------------- Rich Document Preview with Tables -------------------
                doc_path = meta.get("file_path") or meta.get("source") or None
                if doc_path and os.path.exists(doc_path):
                    ext = os.path.splitext(doc_path)[-1].lower()
                    html_content = ""

                    if ext == ".pdf":
                        # PDF: basic text + try extracting tables
                        with pdfplumber.open(doc_path) as pdf:
                            pages_html = []
                            for page in pdf.pages:
                                text = html.escape(page.extract_text() or "").replace("\n", "<br>")
                                tables_html = ""
                                for table in page.extract_tables():
                                    tables_html += "<table border='1' style='border-collapse:collapse;margin:5px;'>"
                                    for row in table:
                                        tables_html += "<tr>" + "".join(f"<td style='padding:3px'>{html.escape(str(cell or ''))}</td>" for cell in row) + "</tr>"
                                    tables_html += "</table>"
                                pages_html.append(text + tables_html)
                            html_content = "<hr>".join(pages_html)

                    elif ext == ".docx":
                        doc = Document(doc_path)
                        html_parts = []
                        for p in doc.paragraphs:
                            text = html.escape(p.text)
                            # Bold/Italic
                            if p.runs:
                                for r in p.runs:
                                    if r.bold:
                                        text = text.replace(html.escape(r.text), f"<b>{html.escape(r.text)}</b>")
                                    if r.italic:
                                        text = text.replace(html.escape(r.text), f"<i>{html.escape(r.text)}</i>")
                            # Bullet / numbered lists
                            if p.style.name.startswith("List"):
                                html_parts.append(f"<li>{text}</li>")
                            else:
                                html_parts.append(f"<p>{text}</p>")
                        # Tables
                        for table in doc.tables:
                            table_html = "<table border='1' style='border-collapse:collapse;margin:5px;'>"
                            for row in table.rows:
                                table_html += "<tr>" + "".join(f"<td style='padding:3px'>{html.escape(cell.text)}</td>" for cell in row.cells) + "</tr>"
                            table_html += "</table>"
                            html_parts.append(table_html)
                        html_content = "".join(html_parts)

                    else:
                        with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
                            html_content = "<p>" + html.escape(f.read()).replace("\n", "<br>") + "</p>"

                    st.markdown(
                        f"<div style='height:500px;overflow-y:scroll;padding:10px;border:1px solid #ccc;border-radius:5px;background-color:#f8f8f8'>{html_content}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.write("Full document not available locally.")


    st.divider()
    st.markdown(
        "> **Tip:** Keep your questions specific (location, climate, growth stage). "
        "You can add more PDFs and click **Add to Index** to expand the knowledge base."
    )

    

if __name__ == "__main__":
    main()
