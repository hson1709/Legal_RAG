from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from config import PARENT_DOCUMENTS, PERSIST_DIRECTORY
from tqdm import tqdm 
import pickle
from typing import List, Dict, Any


metadata_label_map = {
    "dieu": "Điều",
    "muc": "Mục",
    "chuong": "Chương",
    "loai_van_ban": "Loại văn bản",
    "chu_de": "Chủ đề",
    "ma_so": "Mã số",
    "ngay_ban_hanh": "Ngày ban hành",
    "noi_ban_hanh": "Nơi ban hành",
    "co_quan_ban_hanh": "Cơ quan ban hành",
    "can_cu": "Căn cứ"
}

class Retriever:
    def __init__(
        self,
        embedding_model,
        pickle_path = PARENT_DOCUMENTS,
        persist_directory = PERSIST_DIRECTORY,
        collection_name="law_docs",
        chunk_size=1024,
        chunk_overlap=100,
        num_chunks=5
    ):
        self.num_chunks = num_chunks
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vectorstore = Chroma(
            embedding_function=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        self.docstore, self.child_documents = self._load_and_process_documents(pickle_path)
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter
        )

    def _load_and_process_documents(self, pickle_path):
        with open(pickle_path, "rb") as f:
            parent_documents = pickle.load(f)
        docstore = InMemoryStore()
        child_documents = []
        store_items = []
        for parent_doc in tqdm(parent_documents, desc="Tạo docstore & child docs"):
            parent_id = parent_doc.metadata["parent_id"]
            store_items.append((parent_id, parent_doc))
            chunks = self.child_splitter.create_documents(
                [parent_doc.page_content],
                metadatas=[{
                    "parent_id": parent_id,
                    **parent_doc.metadata
                }]
            )
            child_documents.extend(chunks)
        docstore.mset(store_items)
        return docstore, child_documents
    
    
    def _get_unique_parent_docs_from_child_docs(self, child_docs: List) -> Dict[str, Any]:
        """Helper method to get unique parent documents from child documents"""
        unique_parent_docs = {}
        for child_doc in child_docs:
            parent_id = child_doc.metadata.get("parent_id")
            if parent_id and parent_id not in unique_parent_docs:
                parent_doc = self.docstore.mget([parent_id])[0]
                if parent_doc:
                    unique_parent_docs[parent_id] = parent_doc
        return unique_parent_docs
    

    def _format_context(self, parent_docs: List) -> str:
        
        def format_metadata_vietnamese(metadata):
            parts = []
            for key, value in metadata.items():
                if key in ("sourcedoc", "parent_id"):
                    continue
                label = metadata_label_map.get(key, key)
                parts.append(f"{label}: {value}")
            return "(Metadata: " + "; ".join(parts) + ")"
        
        contexts = []
        for i, doc in enumerate(parent_docs, 1):
            content = doc.page_content
            metadata_formatted = format_metadata_vietnamese(doc.metadata)
            contexts.append(f"Tài liệu {i}:\n{content}\n{metadata_formatted}")
        
        return "\n\n".join(contexts)
    

    def retrieve_from_queries(self, queries: List[str], source_filter: str = None) -> str:

        all_unique_parent_docs = {}
        
        for query in queries:
            if not query.strip(): 
                continue

            child_docs = self.vectorstore.similarity_search(query, k=self.num_chunks)

            if source_filter:
                child_docs = [doc for doc in child_docs 
                             if doc.metadata.get("ma_so", "").upper() == source_filter.upper()]

            query_parent_docs = self._get_unique_parent_docs_from_child_docs(child_docs)

            all_unique_parent_docs.update(query_parent_docs)
        
        parent_docs = list(all_unique_parent_docs.values())
        return self._format_context(parent_docs)
    

    def basic_retrieve(self, queries: List[str]) -> str:
        return self.retrieve_from_queries(queries)

    def comparison_retrieve(self, queries: List[str], source_filter: str = None) -> str:
        return self.retrieve_from_queries(queries, source_filter)
    
    def analysis_retrieve(self, queries: List[str]) -> str:
        return self.retrieve_from_queries(queries)


    def get_context(self, query):
        if isinstance(query, str):
            return self.retrieve_from_queries([query])
        elif isinstance(query, list):
            return self.retrieve_from_queries(query)
        else:
            raise ValueError("Query must be string or list of strings")
    

