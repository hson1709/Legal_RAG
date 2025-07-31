from src.components.retriever import BaseRetriever
from typing import List, Any, Union
from underthesea import word_tokenize
import pickle
import re
from rank_bm25 import BM25Okapi
from langsmith import traceable

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

class KeywordRetriever(BaseRetriever):
    def __init__(
        self,
        corpus_path_512="./data/bm25_corpus_512.pkl",
        corpus_path_1024="./data/bm25_corpus_1024.pkl",
        num_chunks=5
    ):
        self.num_chunks = num_chunks
        self.corpus_path_512 = corpus_path_512
        self.corpus_path_1024 = corpus_path_1024
        
        self._load_bm25_corpus()

    def _load_bm25_corpus(self):
        print("Tải BM25 corpus từ file...")

        try:
            with open(self.corpus_path_512, "rb") as f:
                corpus_data_512 = pickle.load(f)

            self.tokenized_corpus_512 = corpus_data_512['tokenized_corpus']
            self.child_docs_512 = corpus_data_512['child_docs']
            self.bm25_512 = BM25Okapi(self.tokenized_corpus_512)

            with open(self.corpus_path_1024, "rb") as f:
                corpus_data_1024 = pickle.load(f)

            self.tokenized_corpus_1024 = corpus_data_1024['tokenized_corpus']
            self.child_docs_1024 = corpus_data_1024['child_docs']
            self.bm25_1024 = BM25Okapi(self.tokenized_corpus_1024)

            print(f"Đã tải thành công BM25 corpus: {len(self.child_docs_512)} docs (512), {len(self.child_docs_1024)} docs (1024)")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Không tìm thấy BM25 corpus file. Vui lòng chạy BM25CorpusBuilder.build_and_save_corpus() trước: {e}")
        except Exception as e:
            raise Exception(f"Lỗi khi tải BM25 corpus: {e}")

    def _preprocess_text(self, text: str) -> List[str]:

        text = text.strip()
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        text = ' '.join(text.split())

        try:
            tokens = word_tokenize(text, format="text")
        except:
            tokens = text.split()
        
        return tokens

    def _get_unique_parent_docs_from_results_with_scores(self, results: List) -> List[tuple]:

        parent_doc_scores = {}
        
        for doc, score in results:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in parent_doc_scores:
                parent_doc = doc.metadata.get("parent_doc")
                if parent_doc:
                    parent_doc_scores[parent_id] = (parent_doc, score)
            elif parent_id in parent_doc_scores:
   
                if score > parent_doc_scores[parent_id][1]:
                    parent_doc_scores[parent_id] = (parent_doc_scores[parent_id][0], score)
 
        sorted_docs = sorted(parent_doc_scores.values(), key=lambda x: x[1], reverse=True)
        return sorted_docs

    def _get_unique_parent_docs_from_results(self, results: List) -> List[Any]:

        seen_parent_ids = set()
        unique_parent_docs = []
        
        for doc, score in results:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parent_ids:
                parent_doc = doc.metadata.get("parent_doc")
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

    def _bm25_search(self, query: str, bm25_index, child_docs) -> List[tuple]:
        k = self.num_chunks

        query_tokens = self._preprocess_text(query)
        scores = bm25_index.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  
                results.append((child_docs[idx], scores[idx]))
        
        return results

    @traceable
    def get_unique_parent_docs_with_scores(self, queries: Union[str, List[str]], use_512: bool = True) -> List[tuple]:

        if isinstance(queries, str):
            queries = [queries]
        
        if use_512:
            bm25_index = self.bm25_512
            child_docs = self.child_docs_512
        else:
            bm25_index = self.bm25_1024
            child_docs = self.child_docs_1024
        
        all_results = []
        
        for query in queries:
            if not query.strip():
                continue
            
            query_results = self._bm25_search(query, bm25_index, child_docs)
            all_results.extend(query_results)
   
        return self._get_unique_parent_docs_from_results_with_scores(all_results)

    def get_unique_parent_docs(self, queries: Union[str, List[str]], use_512: bool = True) -> List[Any]:

        if isinstance(queries, str):
            queries = [queries]
        
        if use_512:
            bm25_index = self.bm25_512
            child_docs = self.child_docs_512
        else:
            bm25_index = self.bm25_1024
            child_docs = self.child_docs_1024
        
        all_results = []
        
        for query in queries:
            if not query.strip():
                continue
            
            query_results = self._bm25_search(query, bm25_index, child_docs)
            all_results.extend(query_results)
   
        return self._get_unique_parent_docs_from_results(all_results)

    def retrieve(self, queries: Union[str, List[str]], use_512: bool = True) -> str:
        parent_docs = self.get_unique_parent_docs(queries, use_512)
        return self._format_context(parent_docs)