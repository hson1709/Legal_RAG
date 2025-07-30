import os
from dotenv import load_dotenv
import torch 

load_dotenv()
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, "model")

RAW_DATA_PATH = (PROJECT_ROOT, "data", "thuvienphapluat_documents.jsonl")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "vector_db")
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
GPT_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-2.5-pro"
PARENT_DOCUMENTS = os.path.join(PROJECT_ROOT, "data", "parent_documents.pkl")
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_CONFIG = {
    "mongodb_uri": MONGODB_URI,
    "mongodb_db_name": "legal_docs",
    "mongodb_collection_name_512": "legal_documents_512",
    "mongodb_collection_name_1024": "legal_documents_1024"
}



__all__ = ["OPENAI_API_KEY","GOOGLE_API_KEY", "RAW_DATA_PATH", "PERSIST_DIRECTORY", "EMBEDDING_MODEL", "RERANKER_MODEL", "GPT_MODEL", "GEMINI_MODEL", "DEVICE", "PARENT_DOCUMENTS", "MONGODB_CONFIG"]
