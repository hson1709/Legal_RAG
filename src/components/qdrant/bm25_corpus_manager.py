import os
import pickle
import logging
from typing import List, Dict, Any
import numpy as np
from qdrant_client.models import SparseVector
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
import re

logger = logging.getLogger(__name__)

class BM25CorpusManager:
    """
    Class độc lập để quản lý BM25 corpus và tạo sparse vectors cho Qdrant
    """
    def __init__(self):
        """
        Khởi tạo BM25 Corpus Manager
        """
        self.bm25 = None
        self.corpus = []  # List[List[str]] - tokenized documents
        self.doc_metadata = []  # List[Dict] - metadata cho từng document
        self.vocabulary = {}  # Dict[str, int] - word -> index mapping
        self.vocab_size = 0
        self.doc_sparse_vectors = {}  # Dict[str, Dict] - doc_id -> sparse vector data

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Tiền xử lý văn bản tiếng Việt
        """
        if not text or not isinstance(text, str):
            return []

        # Chuẩn hóa text
        text = text.strip()
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        text = re.sub(r'\s+', ' ', text)  # Loại bỏ khoảng trắng thừa

        # Loại bỏ ký tự đặc biệt nhưng giữ lại dấu tiếng Việt
        text = text.lower()
        text = ' '.join(text.split())

        if not text:
            return []

        try:
            # Sử dụng underthesea để tokenize tiếng Việt
            tokens = word_tokenize(text, format="text")
            if isinstance(tokens, str):
                tokens = tokens.split()

            # Lọc tokens hợp lệ (loại bỏ từ quá ngắn và số đơn lẻ)
            filtered_tokens = []
            for token in tokens:
                token = token.strip()
                if len(token) >= 2 and not token.isdigit():
                    filtered_tokens.append(token)
                elif len(token) == 1 and token.isalpha():
                    filtered_tokens.append(token)

            return filtered_tokens

        except Exception as e:
            logger.warning(f"Lỗi khi tokenize: {e}, sử dụng split đơn giản")
            tokens = text.split()
            return [token for token in tokens if len(token) >= 2 and token.strip()]

    def build_corpus_from_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Xây dựng corpus từ danh sách documents

        Args:
            documents: List documents với format đã định nghĩa
        """
        logger.info(f"Đang xây dựng corpus từ {len(documents)} documents...")

        self.corpus = []
        self.doc_metadata = []

        for doc in documents:
            # Lấy thông tin cần thiết
            doc_id = doc.get('_id') or doc.get('document_code', '')
            content_text = doc.get('content_text', '')
            parent_id = doc.get('parent_id', '0')

            if not doc_id:
                logger.warning("Document không có _id hoặc document_code, bỏ qua")
                continue

            # Tiền xử lý content_text
            tokens = self._preprocess_text(content_text)

            if tokens:  # Chỉ thêm document có tokens
                self.corpus.append(tokens)
                self.doc_metadata.append({
                    'doc_id': str(doc_id),
                    'parent_id': str(parent_id),
                    'document_title': doc.get('document_title', ''),
                    'document_type': doc.get('document_type', ''),
                    'issuing_authority': doc.get('issuing_authority', ''),
                    'effective_date': doc.get('effective_date', ''),
                    'status': doc.get('status', ''),
                    'token_count': len(tokens)
                })
            else:
                logger.warning(f"Document {doc_id} không có tokens sau xử lý, bỏ qua")

        logger.info(f"Xây dựng thành công corpus với {len(self.corpus)} documents")

        # Xây dựng vocabulary
        self._build_vocabulary()

        # Xây dựng BM25 index
        self._build_bm25_index()

    def _build_vocabulary(self) -> None:
        """
        Xây dựng vocabulary mapping từ corpus
        """
        logger.info("Đang xây dựng vocabulary...")

        vocab_set = set()
        for doc_tokens in self.corpus:
            vocab_set.update(doc_tokens)

        # Sắp xếp vocabulary để đảm bảo tính nhất quán
        sorted_vocab = sorted(list(vocab_set))
        self.vocabulary = {word: idx for idx, word in enumerate(sorted_vocab)}
        self.vocab_size = len(self.vocabulary)

        logger.info(f"Vocabulary có {self.vocab_size} từ duy nhất")

    def _build_bm25_index(self) -> None:
        """
        Xây dựng BM25 index từ corpus
        """
        if not self.corpus:
            logger.error("Corpus rỗng, không thể tạo BM25 index")
            return

        logger.info("Đang xây dựng BM25 index...")

        # Tạo BM25 index với tham số tối ưu cho tiếng Việt
        self.bm25 = BM25Okapi(self.corpus, k1=1.2, b=0.75)

        logger.info("Hoàn thành xây dựng BM25 index")

    def _calculate_bm25_scores_for_doc(self, doc_index: int, k1: float = 1.2, b: float = 0.75) -> Dict[str, float]:
        """
        Tính BM25 scores cho tất cả terms trong một document

        Args:
            doc_index: Index của document trong corpus
            k1, b: BM25 parameters

        Returns:
            Dict mapping term -> BM25 score
        """
        if doc_index >= len(self.corpus):
            return {}

        doc_tokens = self.corpus[doc_index]
        if not doc_tokens:
            return {}

        scores = {}
        doc_len = len(doc_tokens)
        avgdl = sum(len(doc) for doc in self.corpus) / len(self.corpus)
        N = len(self.corpus)

        # Tính score cho từng unique term trong document
        unique_tokens = set(doc_tokens)

        for term in unique_tokens:
            # Term frequency trong document này
            tf = doc_tokens.count(term)

            # Document frequency - số documents chứa term này
            df = sum(1 for doc in self.corpus if term in doc)

            # IDF calculation với smoothing
            idf = np.log((N - df + 0.5) / (df + 0.5))

            # BM25 score
            score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))

            scores[term] = float(score)

        return scores

    def create_sparse_vectors_for_all_docs(self, min_score: float = 0.01) -> Dict[str, Dict[str, Any]]:
        """
        Tạo sparse vectors cho tất cả documents trong corpus

        Args:
            min_score: Threshold tối thiểu cho score

        Returns:
            Dict mapping doc_id -> {sparse_vector: SparseVector, metadata: Dict}
        """
        logger.info("Đang tạo sparse vectors cho tất cả documents...")

        sparse_vectors_data = {}

        for i, metadata in enumerate(self.doc_metadata):
            doc_id = metadata['doc_id']

            # Tính BM25 scores cho document này
            term_scores = self._calculate_bm25_scores_for_doc(i)

            # Chuyển đổi thành format Qdrant sparse vector
            indices = []
            values = []

            for term, score in term_scores.items():
                if score >= min_score and term in self.vocabulary:
                    term_index = self.vocabulary[term]
                    indices.append(term_index)
                    values.append(score)

            # Sắp xếp theo indices (Qdrant yêu cầu)
            if indices:
                sorted_pairs = sorted(zip(indices, values))
                indices, values = zip(*sorted_pairs)
                indices = list(indices)
                values = list(values)

            # Tạo SparseVector
            sparse_vector = SparseVector(indices=indices, values=values)

            # Lưu data
            sparse_vectors_data[doc_id] = {
                'sparse_vector': sparse_vector,
                'metadata': metadata.copy(),
                'vector_stats': {
                    'non_zero_count': len(indices),
                    'max_score': max(values) if values else 0.0,
                    'min_score': min(values) if values else 0.0
                }
            }

            if (i + 1) % 100 == 0:
                logger.info(f"Đã xử lý {i + 1}/{len(self.doc_metadata)} documents")

        self.doc_sparse_vectors = sparse_vectors_data
        logger.info(f"Hoàn thành tạo sparse vectors cho {len(sparse_vectors_data)} documents")

        return sparse_vectors_data

    def create_query_sparse_vector(self, query: str, min_score: float = 0.01) -> SparseVector:
        """
        Tạo sparse vector cho query text (để search)

        Args:
            query: Text cần tìm kiếm
            min_score: Threshold tối thiểu cho score

        Returns:
            SparseVector object của Qdrant
        """
        if not self.bm25 or not self.vocabulary:
            logger.error("BM25 index hoặc vocabulary chưa được xây dựng")
            return SparseVector(indices=[], values=[])

        # Tiền xử lý query
        query_tokens = self._preprocess_text(query)
        if not query_tokens:
            return SparseVector(indices=[], values=[])

        # Tính scores cho query terms
        indices = []
        values = []

        unique_tokens = set(query_tokens)

        for token in unique_tokens:
            if token in self.vocabulary:
                term_index = self.vocabulary[token]

                # TF trong query
                tf = query_tokens.count(token)

                # IDF từ corpus
                df = sum(1 for doc in self.corpus if token in doc)
                if df > 0:
                    idf = np.log(len(self.corpus) / df)
                    score = tf * idf

                    if score >= min_score:
                        indices.append(term_index)
                        values.append(float(score))

        # Sắp xếp theo indices
        if indices:
            sorted_pairs = sorted(zip(indices, values))
            indices, values = zip(*sorted_pairs)
            indices = list(indices)
            values = list(values)

        return SparseVector(indices=indices, values=values)

    def save_sparse_vectors_to_pkl(self, filepath: str) -> None:
        """
        Lưu danh sách sparse vectors và parent_id vào file .pkl

        Args:
            filepath: Đường dẫn file để lưu
        """
        # Tạo list đơn giản chỉ chứa sparse_vector và parent_id
        sparse_vectors_list = []

        for doc_id, data in self.doc_sparse_vectors.items():
            sparse_vector = data['sparse_vector']
            parent_id = data['metadata']['parent_id']

            sparse_vectors_list.append({
                'sparse_vector': sparse_vector,
                'parent_id': parent_id
            })

        # Lưu chỉ list này vào file
        with open(filepath, 'wb') as f:
            pickle.dump(sparse_vectors_list, f)

        logger.info(f"Đã lưu {len(sparse_vectors_list)} sparse vectors vào {filepath}")

    def load_sparse_vectors_from_pkl(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load danh sách sparse vectors từ file .pkl

        Args:
            filepath: Đường dẫn file để load

        Returns:
            List các dict chứa doc_id, sparse_vector, parent_id
        """
        if not os.path.exists(filepath):
            logger.error(f"File {filepath} không tồn tại")
            return []

        with open(filepath, 'rb') as f:
            sparse_vectors_list = pickle.load(f)

        logger.info(f"Đã load {len(sparse_vectors_list)} sparse vectors từ {filepath}")

        return sparse_vectors_list

    def get_qdrant_points_data(self) -> List[Dict[str, Any]]:
        """
        Chuẩn bị data để insert vào Qdrant từ sparse vectors đã tạo

        Returns:
            List các point data với format phù hợp cho Qdrant
        """
        points_data = []

        for doc_id, data in self.doc_sparse_vectors.items():
            sparse_vector = data['sparse_vector']
            parent_id = data['metadata']['parent_id']

            points_data.append({
                'sparse_vector': sparse_vector,
                'payload': {'parent_id': parent_id}
            })

        logger.info(f"Chuẩn bị {len(points_data)} points data cho Qdrant")
        return points_data

    @staticmethod
    def create_qdrant_points_from_pkl(pkl_filepath: str) -> List[Dict[str, Any]]:
        """
        Tạo points data cho Qdrant trực tiếp từ file .pkl

        Args:
            pkl_filepath: Đường dẫn file .pkl chứa sparse vectors

        Returns:
            List các point data để insert vào Qdrant
        """
        if not os.path.exists(pkl_filepath):
            logger.error(f"File {pkl_filepath} không tồn tại")
            return []

        with open(pkl_filepath, 'rb') as f:
            sparse_vectors_list = pickle.load(f)

        points_data = []
        for item in sparse_vectors_list:
            points_data.append({
                'sparse_vector': item['sparse_vector'],
                'payload': {'parent_id': item['parent_id']}
            })

        logger.info(f"Tạo {len(points_data)} points data từ file {pkl_filepath}")
        return points_data

    def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê về corpus và sparse vectors
        """
        if not self.doc_sparse_vectors:
            return {'error': 'Chưa có sparse vectors data'}

        # Thống kê về sparse vectors
        non_zero_counts = []
        max_scores = []
        parent_ids = set()

        for data in self.doc_sparse_vectors.values():
            stats = data.get('vector_stats', {})
            non_zero_counts.append(stats.get('non_zero_count', 0))
            max_scores.append(stats.get('max_score', 0))
            parent_ids.add(data['metadata']['parent_id'])

        return {
            'total_documents': len(self.doc_sparse_vectors),
            'vocabulary_size': self.vocab_size,
            'unique_parent_ids': len(parent_ids),
            'sparse_vector_stats': {
                'avg_non_zero_terms': np.mean(non_zero_counts) if non_zero_counts else 0,
                'max_non_zero_terms': max(non_zero_counts) if non_zero_counts else 0,
                'min_non_zero_terms': min(non_zero_counts) if non_zero_counts else 0,
                'avg_max_score': np.mean(max_scores) if max_scores else 0,
                'global_max_score': max(max_scores) if max_scores else 0
            },
            'corpus_stats': {
                'total_docs': len(self.corpus) if self.corpus else 0,
                'avg_doc_length': sum(len(doc) for doc in self.corpus) / len(self.corpus) if self.corpus else 0,
                'total_tokens': sum(len(doc) for doc in self.corpus) if self.corpus else 0
            } if hasattr(self, 'corpus') else {}
        }