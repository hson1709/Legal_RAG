from langchain.schema import Document
from typing import List, Dict
from src.components.qdrant.bm25_corpus_manager import BM25CorpusManager
from src.components.qdrant.qdrant_index import LawDocumentStorage
from qdrant_client import QdrantClient
from config import QDRANT_API, QDRANT_URL
from src.components.model_loader import load_embedding_model_qdrant


import pickle

with open("./data/docs_json_format_qdrant.pkl", "rb") as f:
    docs_json_format = pickle.load(f)

def create_and_save_parent_documents(docs: List[Dict]) -> List[Document]:

    metadata_field_map = {
    "document_code": "ma_so",
    "document_title": "chu_de",
    "document_type": "loai_van_ban",
    "issuing_authority": "co_quan_ban_hanh",
    "issuing_place": "noi_ban_hanh",
    "effective_date": "ngay_ban_hanh",
    "status": "tinh_trang",
    "parent_id": "parent_id"
    }
    parent_documents = []

    for i, doc in enumerate(docs):
        # Lấy nội dung văn bản
        content_text = doc.get('content_text', '')

        # Tạo metadata với ánh xạ key
        metadata = {
            metadata_field_map[k]: v
            for k, v in doc.items()
            if k in metadata_field_map
        }

        # Tạo Document
        document = Document(page_content=content_text, metadata=metadata)
        parent_documents.append(document)

    return parent_documents

def build_parent_map(parent_docs):

    return {
        doc.metadata.get("parent_id"): {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in parent_docs
        if doc.metadata.get("parent_id") is not None
    }


# Create parent documents
parent_docs = create_and_save_parent_documents(docs_json_format)
parent_map = build_parent_map(parent_docs)

with open("./parent_docs_qdrant.pkl", "wb") as f:
    pickle.dump(parent_map, f)


# Create sparse_vectors
bm25_manager = BM25CorpusManager()
bm25_manager.build_corpus_from_documents(docs_json_format)
sparse_vectors_data = bm25_manager.create_sparse_vectors_for_all_docs(min_score=0.01)
bm25_manager.save_sparse_vectors_to_pkl('./sparse_vectors.pkl')


# Indexing vevtor to Qdrant
loaded_vectors = bm25_manager.load_sparse_vectors_from_pkl('./sparse_vectors.pkl')
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API,
)
embedding_model = load_embedding_model_qdrant()
indexing = LawDocumentStorage(qdrant_client, embedding_model)
indexing.create_collection(enable_sparse_vectors=True)
indexing.add_documents(docs_json_format, './data/sparse_vectors.pkl', batch_size=100)