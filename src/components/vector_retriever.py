from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from config import PARENT_DOCUMENTS, PERSIST_DIRECTORY
from tqdm import tqdm 
import pickle
from typing import List, Any, Union, Dict, Optional
from src.components.retriever import BaseRetriever


metadata_label_map = {
    "ma_so": "Mã số",
    "chu_de": "Chủ đề",
    "loai_van_ban": "Loại văn bản",
    "co_quan_ban_hanh": "Cơ quan ban hành",
    "noi_ban_hanh": "Nơi ban hành",
    "ngay_ban_hanh": "Ngày ban hành",
    "tinh_trang": "Tình trạng",
    "chuong": "Chương",
    "ten_chuong": "Tên chương",
    "muc": "Mục",
    "ten_muc": "Tên mục",
    "ten_dieu": "Tên điều"
}

class VectorRetriever(BaseRetriever):
    def __init__(
        self,
        embedding_model,
        pickle_path="./data/parent_documents_mongo.pkl",
        persist_directory="./vector_stores",
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
    
    def _get_unique_parent_docs_from_child_docs_with_scores(self, child_docs_with_scores: List) -> List[tuple]:
        parent_doc_scores = {}
        
        for child_doc, score in child_docs_with_scores:
            parent_id = child_doc.metadata.get("parent_id")
            if parent_id and parent_id not in parent_doc_scores:
                parent_doc = self.docstore.mget([parent_id])[0]
                if parent_doc:
                    parent_doc_scores[parent_id] = (parent_doc, score)
            elif parent_id in parent_doc_scores:
                # Keep the higher score
                if score > parent_doc_scores[parent_id][1]:
                    parent_doc_scores[parent_id] = (parent_doc_scores[parent_id][0], score)
        
        sorted_docs = sorted(parent_doc_scores.values(), key=lambda x: x[1], reverse=True)
        return sorted_docs

    def _get_unique_parent_docs_from_child_docs(self, child_docs: List) -> List[Any]:
        seen_parent_ids = set()
        unique_parent_docs = []
        
        for child_doc in child_docs:
            parent_id = child_doc.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parent_ids:
                parent_doc = self.docstore.mget([parent_id])[0]
                if parent_doc:
                    unique_parent_docs.append(parent_doc)
                    seen_parent_ids.add(parent_id)
        
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

    def _safe_vector_search_with_score(self, query: str, k: int, where_clause: Optional[Dict] = None):
        """
        Safe vector search that handles where clause conflicts
        
        Args:
            query: Search query
            k: Number of results to return
            where_clause: Optional ChromaDB where clause for filtering
            
        Returns:
            List of (document, score) tuples
        """
        try:
            if where_clause:
                # Try with where clause first
                return self.vectorstore.similarity_search_with_score(
                    query=query, 
                    k=k,
                    filter=where_clause
                )
            else:
                # Normal search without where clause
                return self.vectorstore.similarity_search_with_score(query, k=k)
                
        except Exception as e:
            print(f"Error with where clause in vector search: {e}")
            # Fallback to normal search if where clause fails
            try:
                return self.vectorstore.similarity_search_with_score(query, k=k)
            except Exception as fallback_error:
                print(f"Fallback vector search also failed: {fallback_error}")
                return []

    def _safe_vector_search(self, query: str, k: int, where_clause: Optional[Dict] = None):
        """
        Safe vector search that handles where clause conflicts
        
        Args:
            query: Search query
            k: Number of results to return
            where_clause: Optional ChromaDB where clause for filtering
            
        Returns:
            List of documents
        """
        try:
            if where_clause:
                # Try with where clause first
                return self.vectorstore.similarity_search(
                    query=query, 
                    k=k,
                    filter=where_clause
                )
            else:
                # Normal search without where clause
                return self.vectorstore.similarity_search(query, k=k)
                
        except Exception as e:
            print(f"Error with where clause in vector search: {e}")
            # Fallback to normal search if where clause fails
            try:
                return self.vectorstore.similarity_search(query, k=k)
            except Exception as fallback_error:
                print(f"Fallback vector search also failed: {fallback_error}")
                return []

    def get_unique_parent_docs_with_scores(self, queries: Union[str, List[str]], where_clause: Optional[Dict] = None) -> List[tuple]:
        """
        Get unique parent documents with scores, optionally filtered by where_clause
        
        Args:
            queries: Search query or list of queries
            where_clause: Optional ChromaDB where clause for filtering
            
        Returns:
            List of (parent_doc, score) tuples
        """
        if isinstance(queries, str):
            queries = [queries]
        
        all_child_docs_with_scores = []
        
        for query in queries:
            if not query.strip(): 
                continue
            
            # Use safe search method
            child_docs = self._safe_vector_search_with_score(query, self.num_chunks, where_clause)
            all_child_docs_with_scores.extend(child_docs)
        
        return self._get_unique_parent_docs_from_child_docs_with_scores(all_child_docs_with_scores)

    def get_unique_parent_docs(self, queries: Union[str, List[str]], where_clause: Optional[Dict] = None) -> List[Any]:
        """
        Get unique parent documents, optionally filtered by where_clause
        
        Args:
            queries: Search query or list of queries
            where_clause: Optional ChromaDB where clause for filtering
            
        Returns:
            List of parent documents
        """
        if isinstance(queries, str):
            queries = [queries]
        
        all_child_docs = []
        
        for query in queries:
            if not query.strip(): 
                continue
            
            # Use safe search method
            child_docs = self._safe_vector_search(query, self.num_chunks, where_clause)
            all_child_docs.extend(child_docs)
        
        return self._get_unique_parent_docs_from_child_docs(all_child_docs)

    def retrieve(self, queries: Union[str, List[str]], where_clause: Optional[Dict] = None) -> str:
        """
        Retrieve formatted context, optionally filtered by where_clause
        
        Args:
            queries: Search query or list of queries  
            where_clause: Optional ChromaDB where clause for filtering
            
        Returns:
            Formatted context string
        """
        parent_docs = self.get_unique_parent_docs(queries, where_clause)
        return self._format_context(parent_docs)