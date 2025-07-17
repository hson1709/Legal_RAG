import os
from dotenv import load_dotenv
import torch 

load_dotenv()

os.environ["HF_HOME"] = "D:/legal_rag_project/model"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERSIST_DIRECTORY = "D:/legal_rag_project/vector_db"
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"
LLM_MODEL = "gpt-4o"
PARENT_DOCUMENTS = "D:/legal_rag_project/data/parent_documents.pkl"


__all__ = ["OPENAI_API_KEY", "PERSIST_DIRECTORY", "EMBEDDING_MODEL", "LLM_MODEL", "DEVICE"]
