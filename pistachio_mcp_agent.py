# pistachio_mcp_agent.py
import os
import asyncio
from typing import List
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.schema import TextNode
# from llama_index.embeddings.base import DefaultEmbedding
# from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
import os
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
#from llama_index.core import set_global_embedding, set_global_llm


from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader


# ===================== CONFIG =====================
PERSIST_DIR = "storage"
DATA_DIR = "data"

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))
TOP_K = int(os.getenv("TOP_K", "4"))

# Pistachio-specific keywords for guardrail
ALLOWED_KEYWORDS = [
    "pistachio", "nut", "orchard", "tree", "soil", "irrigation",
    "fertilizer", "harvest", "planting", "pruning", "rootstock"
]

# ===================== HELPER FUNCTIONS =====================
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)

def is_relevant(query: str) -> bool:
    """Guardrail: only allow pistachio-related queries"""
    query_lower = query.lower()
    return any(word in query_lower for word in ALLOWED_KEYWORDS)

# def load_or_build_index() -> VectorStoreIndex:
#     """Load persisted FAISS index or build from DATA_DIR"""
#     if os.path.isdir(PERSIST_DIR) and any(os.scandir(PERSIST_DIR)):
#         storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
#         return load_index_from_storage(storage_context)
#     else:
#         docs = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()
#         index = VectorStoreIndex.from_documents(docs) if docs else VectorStoreIndex([])
#         index.storage_context.persist(persist_dir=PERSIST_DIR)
#         return index

def load_or_build_index() -> VectorStoreIndex:
    """Load persisted FAISS index or build from DATA_DIR"""
    if os.path.isdir(PERSIST_DIR) and any(os.scandir(PERSIST_DIR)):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_context)
    else:
        docs = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()
        index = VectorStoreIndex.from_documents(docs) if docs else VectorStoreIndex([])
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        return index

# def load_or_build_index():
#     # Ensure Settings.embed_model is already configured BEFORE loading
#     if os.path.isdir(PERSIST_DIR) and any(os.scandir(PERSIST_DIR)):
#         storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
#         return load_index_from_storage(storage_context)
#     else:
#         docs = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()
#         if not docs:
#             return VectorStoreIndex([])  # empty index
#         index = VectorStoreIndex.from_documents(docs)
#         index.storage_context.persist(persist_dir=PERSIST_DIR)
#         return index

def append_uploaded_files_to_index(index: VectorStoreIndex, uploaded_files) -> int:
    """Save uploaded files to DATA_DIR, parse, and add to index"""
    if not uploaded_files:
        return 0

    saved_paths: List[str] = []
    for uf in uploaded_files:
        path = os.path.join(DATA_DIR, uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        saved_paths.append(path)

    new_docs = SimpleDirectoryReader(input_files=saved_paths).load_data()
    if not new_docs:
        return 0

    nodes: List[TextNode] = []
    for d in new_docs:
        nodes.append(TextNode(text=d.get_content(), metadata=d.metadata))

    index.insert_nodes(nodes)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return len(new_docs)

# ===================== AGENT INITIALIZATION =====================
# LLM and embeddings
llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

embed_model_name = os.getenv("EMBED_MODEL", "nomic-embed-text")  # 'local' not directly supported
base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

Settings.embed_model = OllamaEmbedding(model_name=embed_model_name, base_url=base_url)

# Load or build index
ensure_dirs()
INDEX = load_or_build_index()

# ===================== MCP-READY ASYNC FUNCTION =====================
async def pistachio_query(query: str) -> str:
    """
    Async function to run a pistachio query.
    Returns answer or guardrail warning.
    """
    if not query.strip():
        return "⚠️ Query cannot be empty."
    
    if not is_relevant(query):
        return "⚠️ Sorry, I can only answer questions about pistachio growing."

    # Use local RAG query engine
    qe = INDEX.as_query_engine(llm=llm, similarity_top_k=TOP_K)
    
    try:
        # Since LlamaIndex queries are blocking, wrap in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: qe.query(query))
        return response.response if hasattr(response, "response") else str(response)
    except Exception as e:
        return f"⚠️ Error running query: {e}"
