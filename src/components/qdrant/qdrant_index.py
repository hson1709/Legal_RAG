import os
import pickle
import logging
from typing import List, Dict, Any
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import CollectionDescription, SparseVector, PointStruct
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import uuid

logger = logging.getLogger(__name__)

class LawDocumentStorage:
    def __init__(self, qdrant_client, embedding_model, collection_name = "law_docs"):
        """
        Khởi tạo kết nối Qdrant và model embedding

        Args:
            qdrant_client: Qdrant client instance
            embedding_model: Model embedding (SentenceTransformer)
        """
        self.client = qdrant_client
        self.collection_name = collection_name

        # Khởi tạo model embedding Vietnamese
        logger.info("Đang tải model embedding...")
        self.embedding_model = embedding_model
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Model đã tải xong. Kích thước vector: {self.vector_size}")

        # Text splitter cho chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        )

        # Mapping metadata fields
        self.metadata_field_map = {
            "document_code": "ma_so",
            "document_title": "chu_de",
            "document_type": "loai_van_ban",
            "issuing_authority": "co_quan_ban_hanh",
            "issuing_place": "noi_ban_hanh",
            "effective_date": "ngay_ban_hanh",
            "status": "tinh_trang",
            "parent_id": "parent_id"
        }

    def create_collection(self, enable_sparse_vectors: bool = True):
        """
        Tạo collection với cấu hình HNSW cho multiple vectors và sparse vectors

        Args:
            enable_sparse_vectors: Có tạo sparse vectors không (mặc định True cho hybrid search)
        """
        try:
            # Xóa collection cũ nếu tồn tại
            existing_collections = self.client.get_collections().collections
            collection_names = [col.name for col in existing_collections]

            if self.collection_name in collection_names:
                logger.info(f"Collection {self.collection_name} đã tồn tại. Đang xóa...")
                self.client.delete_collection(self.collection_name)

            # Cấu hình HNSW parameters
            hnsw_config = models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000,
                max_indexing_threads=0,
                on_disk=False
            )

            # Tạo collection với multiple named vectors
            vectors_config = {
                "title": models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                    hnsw_config=hnsw_config
                ),
                "summary": models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                    hnsw_config=hnsw_config
                ),
                "content": models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                    hnsw_config=hnsw_config
                )
            }

            # Thêm sparse vector config cho hybrid search
            sparse_vectors_config = None
            if enable_sparse_vectors:
                sparse_vectors_config = {
                    "sparse_bm25": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                            datatype=models.Datatype.FLOAT32
                        )
                    )
                }

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=2,
                    indexing_threshold=20000,
                    flush_interval_sec=5
                ),
                wal_config=models.WalConfigDiff(
                    wal_capacity_mb=32,
                    wal_segments_ahead=0
                ),
                shard_number=1,
                replication_factor=1
            )

            sparse_info = " với sparse vectors (hybrid search)" if enable_sparse_vectors else ""
            logger.info(f"Collection '{self.collection_name}' đã được tạo thành công{sparse_info}")

        except Exception as e:
            logger.error(f"Lỗi khi tạo collection: {e}")
            raise

    def _prepare_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chuẩn bị metadata theo mapping đã định nghĩa

        Args:
            document: Document gốc

        Returns:
            Metadata đã được mapping
        """
        metadata = {}

        for original_field, mapped_field in self.metadata_field_map.items():
            if original_field in document and document[original_field] is not None:
                metadata[mapped_field] = document[original_field]

        return metadata

    def _chunk_content(self, content: str) -> List[str]:
        """
        Chia content thành chunks sử dụng RecursiveCharacterTextSplitter

        Args:
            content: Nội dung cần chia

        Returns:
            List các chunks
        """
        if not content or not isinstance(content, str):
            return []

        chunks = self.text_splitter.split_text(content)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def load_sparse_vectors(self, pkl_filepath: str) -> Dict[str, SparseVector]:
        """
        Load sparse vectors từ file .pkl

        Args:
            pkl_filepath: Đường dẫn file .pkl

        Returns:
            Dict mapping parent_id -> sparse_vector
        """
        if not os.path.exists(pkl_filepath):
            logger.error(f"File {pkl_filepath} không tồn tại")
            return {}

        with open(pkl_filepath, 'rb') as f:
            sparse_vectors_list = pickle.load(f)

        # Tạo mapping parent_id -> sparse_vector
        sparse_vectors_dict = {}
        for item in sparse_vectors_list:
            parent_id = item['parent_id']
            sparse_vector = item['sparse_vector']
            sparse_vectors_dict[parent_id] = sparse_vector

        logger.info(f"Đã load {len(sparse_vectors_dict)} sparse vectors từ {pkl_filepath}")
        return sparse_vectors_dict


    logger = logging.getLogger(__name__)

    def add_documents(self, documents: List[Dict[str, Any]], sparse_vectors_pkl: str, batch_size: int = 100):
        """
        Thêm documents vào Qdrant collection với chunking và sparse vectors

        Args:
            documents: List documents cần thêm
            sparse_vectors_pkl: Đường dẫn file .pkl chứa sparse vectors
            batch_size: Kích thước batch để insert
        """
        logger.info(f"Bắt đầu thêm {len(documents)} documents vào collection {self.collection_name}")

        # Load sparse vectors
        sparse_vectors_dict = self.load_sparse_vectors(sparse_vectors_pkl)

        # Chuẩn bị tất cả texts để embedding
        all_titles = []
        all_summaries = []
        all_contents = []
        points_info = []  # Lưu thông tin để tạo points sau

        doc_count = 0
        total_chunks = 0

        for doc in documents:
            # Lấy thông tin cơ bản
            document_title = doc.get('document_title', '')
            content_summary = doc.get('content_summary', '')
            content_text = doc.get('content_text', '')
            parent_id = str(doc.get('parent_id', ''))

            # Skip document nếu không có sparse vector
            if parent_id not in sparse_vectors_dict:
                logger.warning(f"Không tìm thấy sparse vector cho parent_id {parent_id}, bỏ qua document")
                continue

            # Chuẩn bị metadata
            metadata = self._prepare_metadata(doc)

            # Chunk content
            content_chunks = self._chunk_content(content_text)
            if not content_chunks:
                logger.warning(f"Document {parent_id} không có content chunks, bỏ qua")
                continue

            # Lưu thông tin cho từng chunk (point)
            for chunk in content_chunks:
                all_titles.append(document_title)
                all_summaries.append(content_summary)
                all_contents.append(chunk)
                points_info.append({
                    'metadata': metadata,
                    'sparse_vector': sparse_vectors_dict[parent_id]
                })

            doc_count += 1
            total_chunks += len(content_chunks)

            if doc_count % 10 == 0:
                logger.info(f"Đã chuẩn bị {doc_count}/{len(documents)} documents, tổng {total_chunks} chunks")

        logger.info(f"Hoàn thành chuẩn bị. Tổng cộng: {doc_count} documents, {total_chunks} chunks")

        if not all_titles:
            logger.warning("Không có documents hợp lệ để thêm vào")
            return

        # Tạo embeddings
        logger.info("Đang tạo embeddings...")
        title_embeddings = self.embedding_model.encode(all_titles, show_progress_bar=True, batch_size=32)
        summary_embeddings = self.embedding_model.encode(all_summaries, show_progress_bar=True, batch_size=32)
        content_embeddings = self.embedding_model.encode(all_contents, show_progress_bar=True, batch_size=32)

        # Tạo và insert points theo batch với tqdm
        logger.info(f"Đang insert {total_chunks} points vào Qdrant...")

        for i in tqdm(range(0, total_chunks, batch_size), desc="Indexing to Qdrant"):
            batch_points = []
            for j in range(i, min(i + batch_size, total_chunks)):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "title": title_embeddings[j].tolist(),
                        "summary": summary_embeddings[j].tolist(),
                        "content": content_embeddings[j].tolist(),
                        "sparse_bm25": points_info[j]['sparse_vector']
                    },
                    payload=points_info[j]['metadata']
                )
                batch_points.append(point)

            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
            except Exception as e:
                logger.error(f"Lỗi khi insert batch từ index {i}: {e}")
                raise

        logger.info(f"Hoàn thành thêm {total_chunks} points từ {doc_count} documents vào collection '{self.collection_name}'")


    def get_collection_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về collection

        Returns:
            Thông tin collection
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'collection_name': self.collection_name,
                'vectors_count': collection_info.vectors_count,
                'indexed_vectors_count': collection_info.indexed_vectors_count,
                'points_count': collection_info.points_count,
                'segments_count': collection_info.segments_count,
                'status': collection_info.status,
                'optimizer_status': collection_info.optimizer_status,
                'config': {
                    'vector_size': self.vector_size,
                    'distance': 'COSINE',
                    'has_sparse_vectors': True,
                    'chunk_size': 1024,
                    'chunk_overlap': 100
                }
            }
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin collection: {e}")
            return {'error': str(e)}

    def search_documents(self, query: str, search_type: str = "hybrid", limit: int = 10,
                        score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Tìm kiếm documents trong collection

        Args:
            query: Câu truy vấn
            search_type: Loại tìm kiếm ("semantic", "sparse", "hybrid")
            limit: Số lượng kết quả trả về
            score_threshold: Ngưỡng điểm số tối thiểu

        Returns:
            List kết quả tìm kiếm
        """
        try:
            if search_type == "semantic":
                # Tìm kiếm semantic với dense vectors
                query_vector = self.embedding_model.encode([query])[0]

                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=("content", query_vector.tolist()),
                    limit=limit,
                    score_threshold=score_threshold
                )

            elif search_type == "sparse":
                # Tìm kiếm với sparse vectors (cần implement BM25 query vector)
                logger.warning("Sparse search chưa được implement đầy đủ")
                return []

            elif search_type == "hybrid":
                # Hybrid search kết hợp dense và sparse
                query_vector = self.embedding_model.encode([query])[0]

                # Ở đây có thể implement hybrid search phức tạp hơn
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=("content", query_vector.tolist()),
                    limit=limit,
                    score_threshold=score_threshold
                )
            else:
                raise ValueError(f"Unsupported search_type: {search_type}")

            # Format kết quả
            results = []
            for hit in search_result:
                result = {
                    'id': hit.id,
                    'score': hit.score,
                    'payload': hit.payload
                }
                results.append(result)

            logger.info(f"Tìm thấy {len(results)} kết quả cho query: '{query}'")
            return results

        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm: {e}")
            return []

    def delete_collection(self):
        """
        Xóa collection
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Đã xóa collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Lỗi khi xóa collection: {e}")
            raise