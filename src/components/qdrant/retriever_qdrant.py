from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from typing import List, Dict, Tuple, Optional, Set
from src.components.qdrant.bm25_corpus_manager import BM25CorpusManager

class SearchManager:
    """
    Class quản lý hybrid search kết hợp dense và sparse vectors với multi-vector support
    """

    def __init__(self,
                 embedding_model,
                 parent_docs,
                 qdrant_client: QdrantClient,
                 collection_name: str,
                 bm25_manager: 'BM25CorpusManager' = None
                 ):
        """
        Khởi tạo Hybrid Search Manager

        Args:
            qdrant_client: Qdrant client
            collection_name: Tên collection
            bm25_manager: Instance của BM25CorpusManager (nếu sử dụng custom BM25)
            embedding_model: Model cho dense embedding
            parent_docs: Dict chứa parent documents
        """
        self.client = qdrant_client
        self.collection_name = collection_name
        self.bm25_manager = bm25_manager
        self.parent_docs = parent_docs
        self.embedding_model = embedding_model


    def create_query_vectors(self, query: str) -> Tuple[List[float], SparseVector]:
        """
        Tạo cả dense và sparse vectors cho query

        Args:
            query: Query text

        Returns:
            Tuple of (dense_vector, sparse_vector)
        """

        dense_vector = self.embedding_model.encode([query])[0]

        sparse_vector = self.bm25_manager.create_query_sparse_vector(query)


        return dense_vector, sparse_vector

    def _extract_parent_ids_from_results(self, results: List[Dict]) -> Set[str]:
        """
        Trích xuất unique parent_ids từ search results
        
        Args:
            results: List kết quả search
            
        Returns:
            Set các parent_id unique
        """
        parent_ids = set()
        for result in results:
            if 'payload' in result and 'parent_id' in result['payload']:
                parent_ids.add(result['payload']['parent_id'])
        return parent_ids

    def _format_context(self, unique_parent_ids: Set[str]) -> str:
        """
        Format context từ set/list unique_parent_ids dựa vào parent_docs
        
        Args:
            unique_parent_ids: Set các parent_id unique
            
        Returns:
            Formatted context string
        """
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

        def format_metadata_vietnamese(metadata):
            parts = []
            for key, value in metadata.items():
                if key == "parent_id":
                    continue
                label = metadata_label_map.get(key, key)
                parts.append(f"{label}: {value}")
            return "(Metadata: " + "; ".join(parts) + ")"

        contexts = []
        for i, pid in enumerate(sorted(unique_parent_ids), 1):
            if pid not in self.parent_docs:
                continue
            entry = self.parent_docs[pid]
            content = entry["content"]
            metadata_formatted = format_metadata_vietnamese(entry["metadata"])
            contexts.append(f"Tài liệu {i}:\n{content}\n{metadata_formatted}")

        return "\n\n".join(contexts)

    def search_single_vector_context(self,
                                   query: str,
                                   vector_name: str,
                                   limit: int = 10,
                                   score_threshold: Optional[float] = None) -> str:
        """
        Tìm kiếm sử dụng một vector cụ thể và trả về context

        Args:
            query: Query text
            vector_name: Tên vector ("content", "title", "summary", "sparse_bm25")
            limit: Số kết quả trả về
            score_threshold: Ngưỡng điểm tối thiểu

        Returns:
            Formatted context string
        """
        if vector_name == "sparse_bm25":
            # Sử dụng sparse vector
            _, sparse_vector = self.create_query_vectors(query)

            search_params = {
                "collection_name": self.collection_name,
                "query": sparse_vector,
                "using": vector_name,
                "limit": limit,
                "with_payload": True,
                "with_vectors": False
            }

            if score_threshold:
                search_params["score_threshold"] = score_threshold

            results = self.client.query_points(**search_params)

        else:
            # Sử dụng dense vector cho content, title, summary
            dense_vector, _ = self.create_query_vectors(query)

            search_params = {
                "collection_name": self.collection_name,
                "query": dense_vector.tolist(),
                "using": vector_name,
                "limit": limit,
                "with_payload": True,
                "with_vectors": False
            }

            if score_threshold:
                search_params["score_threshold"] = score_threshold

            results = self.client.query_points(**search_params)

        # Format results để extract parent_ids
        formatted_results = [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
                "vector_used": vector_name
            }
            for point in results.points
        ]

        # Extract parent_ids và format context
        parent_ids = self._extract_parent_ids_from_results(formatted_results)
        return self._format_context(parent_ids)

    def multi_vector_search_context(self,
                                  query: str,
                                  vector_names: List[str] = ["content", "title", "summary"],
                                  limit: int = 10,
                                  score_threshold: Optional[float] = None) -> str:
        """
        Tìm kiếm trên nhiều vectors và trả về context tổng hợp (không trùng văn bản gốc)

        Args:
            query: Query text
            vector_names: List tên các vectors cần search
            limit: Số kết quả cho mỗi vector
            score_threshold: Ngưỡng điểm tối thiểu

        Returns:
            Formatted context string tổng hợp từ tất cả vectors
        """
        all_parent_ids = set()

        for vector_name in vector_names:
            try:
                # Lấy raw results thay vì context
                if vector_name == "sparse_bm25":
                    _, sparse_vector = self.create_query_vectors(query)
                    search_params = {
                        "collection_name": self.collection_name,
                        "query": sparse_vector,
                        "using": vector_name,
                        "limit": limit,
                        "with_payload": True,
                        "with_vectors": False
                    }
                    if score_threshold:
                        search_params["score_threshold"] = score_threshold
                    results = self.client.query_points(**search_params)
                else:
                    dense_vector, _ = self.create_query_vectors(query)
                    search_params = {
                        "collection_name": self.collection_name,
                        "query": dense_vector.tolist(),
                        "using": vector_name,
                        "limit": limit,
                        "with_payload": True,
                        "with_vectors": False
                    }
                    if score_threshold:
                        search_params["score_threshold"] = score_threshold
                    results = self.client.query_points(**search_params)

                # Format results và extract parent_ids
                formatted_results = [
                    {
                        "id": point.id,
                        "score": point.score,
                        "payload": point.payload,
                        "vector_used": vector_name
                    }
                    for point in results.points
                ]

                # Thêm parent_ids từ vector này vào tập hợp chung
                vector_parent_ids = self._extract_parent_ids_from_results(formatted_results)
                all_parent_ids.update(vector_parent_ids)

            except Exception as e:
                print(f"Lỗi khi tìm kiếm vector {vector_name}: {e}")
                continue

        # Format context từ tất cả unique parent_ids
        return self._format_context(all_parent_ids)

    def weighted_multi_vector_search_context(self,
                                            query: str,
                                            vector_weights: Dict[str, float] = None,
                                            limit: int = 10,
                                            search_limit: int = 20,
                                            score_threshold: Optional[float] = None) -> str:
        """
        Tìm kiếm kết hợp nhiều vectors với trọng số và trả về context

        Args:
            query: Query text
            vector_weights: Dict mapping vector_name -> weight
            limit: Số kết quả cuối cùng
            search_limit: Số kết quả tìm kiếm cho mỗi vector
            score_threshold: Ngưỡng điểm tối thiểu

        Returns:
            Formatted context string từ kết quả cuối cùng
        """
        # Default weights
        if vector_weights is None:
            vector_weights = {
                "content": 0.5,
                "title": 0.3,
                "summary": 0.2
            }

        # Tìm kiếm trên từng vector
        all_results = {}
        max_scores = {}

        for vector_name, weight in vector_weights.items():
            if weight > 0:  # Chỉ search vector có trọng số > 0
                # Get raw results (không dùng context method)
                if vector_name == "sparse_bm25":
                    _, sparse_vector = self.create_query_vectors(query)
                    search_params = {
                        "collection_name": self.collection_name,
                        "query": sparse_vector,
                        "using": vector_name,
                        "limit": search_limit,
                        "with_payload": True,
                        "with_vectors": False
                    }
                    if score_threshold:
                        search_params["score_threshold"] = score_threshold
                    results = self.client.query_points(**search_params)
                else:
                    dense_vector, _ = self.create_query_vectors(query)
                    search_params = {
                        "collection_name": self.collection_name,
                        "query": dense_vector.tolist(),
                        "using": vector_name,
                        "limit": search_limit,
                        "with_payload": True,
                        "with_vectors": False
                    }
                    if score_threshold:
                        search_params["score_threshold"] = score_threshold
                    results = self.client.query_points(**search_params)

                vector_results = [
                    {
                        "id": point.id,
                        "score": point.score,
                        "payload": point.payload,
                        "vector_used": vector_name
                    }
                    for point in results.points
                ]

                all_results[vector_name] = vector_results

                # Tìm max score để chuẩn hóa
                if vector_results:
                    max_scores[vector_name] = max(r["score"] for r in vector_results)
                else:
                    max_scores[vector_name] = 1.0

        # Kết hợp kết quả
        combined_scores = {}

        for vector_name, results in all_results.items():
            weight = vector_weights[vector_name]
            max_score = max_scores[vector_name]

            for result in results:
                doc_id = result["id"]
                normalized_score = result["score"] / max_score
                weighted_score = weight * normalized_score

                if doc_id not in combined_scores:
                    combined_scores[doc_id] = {
                        "total_score": 0.0,
                        "payload": result["payload"]
                    }

                combined_scores[doc_id]["total_score"] += weighted_score

        # Tạo kết quả cuối cùng
        final_results = []
        for doc_id, data in combined_scores.items():
            final_results.append({
                "id": doc_id,
                "score": data["total_score"],
                "payload": data["payload"]
            })

        # Sắp xếp theo total score và lấy top results
        final_results.sort(key=lambda x: x["score"], reverse=True)
        final_results = final_results[:limit]

        # Extract parent_ids từ final results và format context
        parent_ids = self._extract_parent_ids_from_results(final_results)
        return self._format_context(parent_ids)

    def hybrid_search_with_sparse_context(self,
                                        query: str,
                                        dense_vectors: List[str] = ["content"],
                                        sparse_weight: float = 0.3,
                                        dense_weights: Dict[str, float] = None,
                                        limit: int = 10,
                                        search_limit: int = 20,
                                        score_threshold: Optional[float] = None) -> str:
        """
        Hybrid search kết hợp sparse vector và multiple dense vectors, trả về context

        Args:
            query: Query text
            dense_vectors: List tên các dense vectors
            sparse_weight: Trọng số cho sparse vector
            dense_weights: Dict trọng số cho từng dense vector
            limit: Số kết quả cuối cùng
            search_limit: Số kết quả search cho mỗi vector
            score_threshold: Ngưỡng điểm tối thiểu

        Returns:
            Formatted context string
        """
        # Default dense weights
        if dense_weights is None:
            total_dense_weight = 1.0 - sparse_weight
            weight_per_vector = total_dense_weight / len(dense_vectors)
            dense_weights = {v: weight_per_vector for v in dense_vectors}

        # Kết hợp sparse weight vào weights dict
        all_weights = dense_weights.copy()
        all_weights["sparse_bm25"] = sparse_weight

        # Sử dụng weighted_multi_vector_search_context với sparse vector
        return self.weighted_multi_vector_search_context(
            query=query,
            vector_weights=all_weights,
            limit=limit,
            search_limit=search_limit,
            score_threshold=score_threshold
        )

    def advanced_hybrid_search_context(self,
                                     query: str,
                                     algorithm: str = "rrf",
                                     k: int = 60,
                                     alpha: float = 0.5,
                                     limit: int = 10) -> str:
        """
        Advanced hybrid search với các thuật toán kết hợp khác nhau, trả về context

        Args:
            query: Query text
            algorithm: "rrf" (Reciprocal Rank Fusion) hoặc "convex" (Convex Combination)
            k: Parameter cho RRF (default 60)
            alpha: Weight cho convex combination (0-1)
            limit: Số kết quả trả về

        Returns:
            Formatted context string
        """
        # Lấy kết quả từ dense và sparse search
        dense_results = self._get_raw_search_results(query, "content", limit*2)
        sparse_results = self._get_raw_search_results(query, "sparse_bm25", limit*2)

        if algorithm == "rrf":
            final_results = self._reciprocal_rank_fusion(dense_results, sparse_results, k, limit)
        elif algorithm == "convex":
            final_results = self._convex_combination(dense_results, sparse_results, alpha, limit)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Extract parent_ids và format context
        parent_ids = self._extract_parent_ids_from_results(final_results)
        return self._format_context(parent_ids)

    def _get_raw_search_results(self, query: str, vector_name: str, limit: int) -> List[Dict]:
        """
        Helper method để lấy raw search results
        """
        if vector_name == "sparse_bm25":
            _, sparse_vector = self.create_query_vectors(query)
            search_params = {
                "collection_name": self.collection_name,
                "query": sparse_vector,
                "using": vector_name,
                "limit": limit,
                "with_payload": True,
                "with_vectors": False
            }
            results = self.client.query_points(**search_params)
        else:
            dense_vector, _ = self.create_query_vectors(query)
            search_params = {
                "collection_name": self.collection_name,
                "query": dense_vector.tolist(),
                "using": vector_name,
                "limit": limit,
                "with_payload": True,
                "with_vectors": False
            }
            results = self.client.query_points(**search_params)

        return [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
                "vector_used": vector_name
            }
            for point in results.points
        ]

    def _reciprocal_rank_fusion(self, dense_results: List[Dict], sparse_results: List[Dict],
                               k: int = 60, limit: int = 10) -> List[Dict]:
        """
        Reciprocal Rank Fusion algorithm
        """
        rrf_scores = {}
        all_docs = {}

        # Tính RRF scores cho dense results
        for rank, result in enumerate(dense_results):
            doc_id = result["id"]
            rrf_score = 1.0 / (k + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
            all_docs[doc_id] = result

        # Tính RRF scores cho sparse results
        for rank, result in enumerate(sparse_results):
            doc_id = result["id"]
            rrf_score = 1.0 / (k + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
            if doc_id not in all_docs:
                all_docs[doc_id] = result

        # Sắp xếp theo RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, rrf_score in sorted_docs[:limit]:
            result = all_docs[doc_id].copy()
            result["rrf_score"] = rrf_score
            final_results.append(result)

        return final_results

    def _convex_combination(self, dense_results: List[Dict], sparse_results: List[Dict],
                           alpha: float = 0.5, limit: int = 10) -> List[Dict]:
        """
        Convex combination: alpha * dense_score + (1-alpha) * sparse_score
        """
        # Chuẩn hóa scores
        if dense_results:
            max_dense_score = max(r["score"] for r in dense_results)
            for r in dense_results:
                r["normalized_dense_score"] = r["score"] / max_dense_score

        if sparse_results:
            max_sparse_score = max(r["score"] for r in sparse_results)
            for r in sparse_results:
                r["normalized_sparse_score"] = r["score"] / max_sparse_score

        # Kết hợp scores
        combined_scores = {}
        all_docs = {}

        for result in dense_results:
            doc_id = result["id"]
            combined_scores[doc_id] = alpha * result["normalized_dense_score"]
            all_docs[doc_id] = result

        for result in sparse_results:
            doc_id = result["id"]
            sparse_contribution = (1 - alpha) * result["normalized_sparse_score"]
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + sparse_contribution
            if doc_id not in all_docs:
                all_docs[doc_id] = result

        # Sắp xếp theo combined score
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, combined_score in sorted_docs[:limit]:
            result = all_docs[doc_id].copy()
            result["combined_score"] = combined_score
            final_results.append(result)

        return final_results