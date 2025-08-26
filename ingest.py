# ingest.py
import os
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

def main():
    load_dotenv()

    llm_model = os.getenv("LLM_MODEL", "llama3")
    embed_model_name = os.getenv("EMBED_MODEL", "nomic-embed-text")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
    chunk_size = int(os.getenv("CHUNK_SIZE", "1024"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "128"))

    # Configure global settings for LlamaIndex
    Settings.llm = Ollama(model=llm_model, base_url=base_url, request_timeout=120)
    Settings.embed_model = OllamaEmbedding(model_name=embed_model_name, base_url=base_url)
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Load docs
    docs_dir = "data"
    if not os.path.isdir(docs_dir):
        raise SystemExit(f"Create a 'data/' folder and place your pistachio docs there. Missing: {docs_dir}")

    print(f"Loading documents from ./{docs_dir} ...")
    documents = SimpleDirectoryReader(docs_dir, recursive=True).load_data()
    if not documents:
        raise SystemExit("No documents found. Add files to ./data and re-run.")

    print(f"Loaded {len(documents)} documents. Building index...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    # Persist to disk
    persist_dir = "storage"
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index.storage_context.persist(persist_dir=persist_dir)

    print(f"Index built and persisted to ./{persist_dir}")

if __name__ == "__main__":
    main()
