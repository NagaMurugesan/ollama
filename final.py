import streamlit as st
from PIL import Image
from src.chat import main_ui
st.set_page_config(layout="wide")
PERSIST_DIR = "storage"
DATA_DIR = "data"
import nltk
import os

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
from dotenv import load_dotenv


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

    print(system_prompt)
    print(llm_model)
    print(embed_model_name)

    # Configure global settings
    Settings.llm = Ollama(model=llm_model, base_url=base_url, request_timeout=120, system_prompt=system_prompt)
    Settings.embed_model = OllamaEmbedding(model_name=embed_model_name, base_url=base_url)
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


import html
# Download stopwords (only needs to be done once)
# Ensure required nltk resources exist
for resource in ["stopwords", "punkt", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download("wordnet")



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

# Add custom CSS to set the zoom level to 90%
st.markdown(
    """
    <style>
        body {
            zoom: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# Adding (css)stye to application
with open('style/final.css') as f:
    st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
    

# Adding company logo
# imcol1, imcol2, imcol3, imcol4, imcol5 = st.columns((3,4,.5,4,3))
imcol1, imcol2, imcol5 = st.columns((5,3.5,5))

with imcol2:
    st.write("# ")
    st.image('image/default_logo.png') 

# with imcol4:
#     st.write("# ")
#     st.write("# ")

st.markdown("<p style='text-align: center; color: black; font-size:25px;'><span style='font-weight: bold'></span>Conversational AI with a Knowledge Base (FedBotics)</p>", unsafe_allow_html=True)
st.markdown("<hr style=height:2.5px;margin-top:0px;width:100%;background-color:#5480cb;>",unsafe_allow_html=True)

ensure_dirs()
configure_llamaindex()

# if uploaded and add_btn:
#         with st.spinner("Indexing uploaded files..."):
#             added = append_uploaded_files_to_index(st.session_state.index, uploaded)
#         st.success(f"Indexed {added} new document(s).")


# def append_uploaded_files_to_index(index: VectorStoreIndex, uploaded_files) -> int:
#     """Save uploaded files to data/, parse, and add to index incrementally."""
#     if not uploaded_files:
#         return 0

#     saved_paths: List[str] = []
#     for uf in uploaded_files:
#         # Save to data/
#         path = os.path.join(DATA_DIR, uf.name)
#         with open(path, "wb") as f:
#             f.write(uf.getbuffer())
#         saved_paths.append(path)

#     # Read the new docs and upsert
#     new_docs = SimpleDirectoryReader(input_files=saved_paths).load_data()
#     if not new_docs:
#         return 0

#     # Convert to Nodes and insert
#     nodes: List[TextNode] = []
#     for d in new_docs:
#         nodes.append(TextNode(text=d.get_content(), metadata=d.metadata))

#     index.insert_nodes(nodes)
#     index.storage_context.persist(persist_dir=PERSIST_DIR)
#     return len(new_docs)        
    
#---------Side bar-------#
with st.sidebar:


    st.markdown("<p style='text-align: center; Black: ; font-size:25px;'><span style='font-weight: bold; font-family: century-gothic';></span>Solutions Scope</p>", unsafe_allow_html=True)
    # vAR_AI_application = st.selectbox("",['Lab-6'],key='application')
    # selected = st.selectbox("",['User',"Logout"],key='text')
    vAR_LLM_model = st.selectbox("",['GEN AI Models',"gpt-3.5-turbo-16k-0613","gpt-4-0314","gpt-3.5-turbo-16k","gpt-3.5-turbo-1106","gpt-4-0613","gpt-4-0314"],key='text_llmmodel')
    vAR_LLM_framework = st.selectbox("",['Framework',"Langchain"],key='text_framework')

    # vAR_Library = st.selectbox("",["Library Used","Streamlit","Image","Pandas","openAI"],key='text1')
    vAR_Gcp_cloud = st.selectbox("",
                    ["Cloud Services","VM Instance","Computer Engine","Cloud Storage"],key='text2')
    st.markdown("#### ")
    href = """<form action="#">
            <input type="submit" value="Clear/Reset"/>
            </form>"""
    st.sidebar.markdown(href, unsafe_allow_html=True)
    st.markdown("# ")
    st.markdown("<p style='text-align: center; color: Black; font-size:20px;'>Build & Deployed on<span style='font-weight: bold'></span></p>", unsafe_allow_html=True)
    #t.write("Index Folder:")

    # st.divider()
    # uploaded = st.file_uploader(
    #     "➕ Add more documents",
    #     type=["pdf", "txt", "md", "docx"],
    #     accept_multiple_files=True
    # )
    # add_btn = st.button("Add to Index")

  
    s2,s3,s4=st.columns((4,4,4))
    
    # with s2:    
    #     st.markdown("### ")
    #     st.image("image/oie_png.png")
    with s3:
        st.markdown("### ")
        st.image('image/aws.png')
    # with s4:    
    #     st.markdown("### ")
    #     st.image("image/AzureCloud_img.png")

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

try:
    # if vAR_AI_application == "Lab-6":
    main_ui()
except BaseException as e:
    st.error("An error occurred. Kindly contact the technical support team.")
