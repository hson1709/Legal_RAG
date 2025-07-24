from src.components.retriever import BaseRetriever
from typing import List, Any, Union
from src.components.vector_retriever import VectorRetriever
from src.components.keyword_retriever import KeywordRetriever
import math

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

class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        embedding_model,
        reranker,
        pickle_path="./data/parent_documents.pkl",
        persist_directory="./vector_stores",
        collection_name="law_docs",
        corpus_path_512="./data/bm25_corpus_512.pkl",
        corpus_path_1024="./data/bm25_corpus_1024.pkl",
        chunk_size=1024,  
        chunk_overlap=100,
        vector_num_chunks=10,
        keyword_num_chunks=10,
        hybrid_num_chunks=5,
        num_final_docs = 5,
        vector_search_weight=0.6,  
    ):
        if chunk_size not in [512, 1024]:
            raise ValueError("chunk_size must be either 512 or 1024")
        
        self.chunk_size = chunk_size
        self.hybrid_num_chunks = hybrid_num_chunks
        self.num_final_docs = num_final_docs
        self.vector_search_weight = vector_search_weight
        self.reranker = reranker

        self.vector_retriever = VectorRetriever(
            embedding_model=embedding_model,
            pickle_path=pickle_path,
            persist_directory=persist_directory,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_chunks=vector_num_chunks
        )
        
        self.keyword_retriever = KeywordRetriever(
            corpus_path_512=corpus_path_512,
            corpus_path_1024=corpus_path_1024,
            num_chunks=keyword_num_chunks
        )

    def _weighted_merge_and_select_docs(
        self, 
        vector_docs_with_scores: List[tuple], 
        keyword_docs_with_scores: List[tuple]
    ) -> List[Any]:
  
        vector_count = math.ceil(self.hybrid_num_chunks * self.vector_search_weight)
        keyword_count = self.hybrid_num_chunks - vector_count

        vector_count = min(vector_count, len(vector_docs_with_scores))
        keyword_count = min(keyword_count, len(keyword_docs_with_scores))

        selected_vector_docs = [doc for doc, score in vector_docs_with_scores[:vector_count]]
        selected_keyword_docs = [doc for doc, score in keyword_docs_with_scores[:keyword_count]]

        unique_docs = {}

        for doc in selected_vector_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in unique_docs:
                unique_docs[parent_id] = doc
 
        for doc in selected_keyword_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in unique_docs:
                unique_docs[parent_id] = doc
        
        final_docs = list(unique_docs.values())

        if len(final_docs) < self.hybrid_num_chunks:
            remaining_needed = self.hybrid_num_chunks - len(final_docs)

            all_remaining_docs = []
            for doc, score in vector_docs_with_scores[vector_count:]:
                parent_id = doc.metadata.get("parent_id")
                if parent_id not in unique_docs:
                    all_remaining_docs.append(doc)
                    
            for doc, score in keyword_docs_with_scores[keyword_count:]:
                parent_id = doc.metadata.get("parent_id")
                if parent_id not in unique_docs:
                    all_remaining_docs.append(doc)
            
            final_docs.extend(all_remaining_docs[:remaining_needed])
        
        return final_docs[:self.hybrid_num_chunks]

    def _format_context(self, parent_docs: List[Any]) -> str:
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


    def retrieve(self, query: Union[str, List[str]]) -> str:
        if isinstance(query, str):
            queries = [query]
        elif isinstance(query, list):
            queries = query
        else:
            raise ValueError("Query must be a string or list of strings")
        
        def write_log(message: str, filepath: str = "D:/legal_rag_project/debug_retriever_log.txt"):
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(f"{message}\n\n")


        write_log(f"[INPUT QUERY]\n{queries}")

        # Vector retrieval
        vector_docs_with_scores = self.vector_retriever.get_unique_parent_docs_with_scores(queries)
        vector_log = "\n\n".join(
            f"[Vector] Score: {score:.4f} | ID: {doc.metadata.get('parent_id')} | Content: {doc.page_content}"
            for doc, score in vector_docs_with_scores
        )
        write_log(f"[VECTOR RETRIEVER]\n{vector_log}")

        # Keyword retrieval
        use_512 = (self.chunk_size == 512)
        keyword_docs_with_scores = self.keyword_retriever.get_unique_parent_docs_with_scores(queries, use_512=use_512)
        keyword_log = "\n\n".join(
            f"[Keyword] Score: {score:.4f} | ID: {doc.metadata.get('parent_id')} | Content: {doc.page_content}"
            for doc, score in keyword_docs_with_scores
        )
        write_log(f"[KEYWORD RETRIEVER]\n{keyword_log}")

        # Hybrid merging
        merged_docs = self._weighted_merge_and_select_docs(vector_docs_with_scores, keyword_docs_with_scores)
        merged_log = "\n\n".join(
            f"[Hybrid] ID: {doc.metadata.get('parent_id')} | Content: {doc.page_content}"
            for doc in merged_docs
        )
        write_log(f"[HYBRID MERGE RESULT]\n{merged_log}")

        # Reranking
        reranked_docs_with_scores = self.reranker.rerank_documents(queries, merged_docs)
        rerank_log = "\n\n".join(
            f"[Reranker] Final Score: {score:.4f} | ID: {doc.metadata.get('parent_id')} | Content: {doc.page_content}"
            for doc, score in reranked_docs_with_scores
        )
        write_log(f"[RERANKER RESULTS]\n{rerank_log}")

        # Final output
        reranked_docs = [doc for doc, _ in reranked_docs_with_scores]
        final_docs = reranked_docs[:self.num_final_docs]
        final_log = "\n\n".join(
            f"[FINAL SELECTED DOC] ID: {doc.metadata.get('parent_id')} | Content: {doc.page_content}"
            for doc in final_docs
        )
        write_log(f"[FINAL OUTPUT]\n{final_log}")

        return self._format_context(final_docs)




