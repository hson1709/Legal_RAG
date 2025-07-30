from typing import List, Any, Union, Dict, Optional
from src.components.generator import Generator
from src.components.intent_classifier import IntentClassifier
from src.components.filter_extractor import FilterExtractor

class RAGPipeline:

    def __init__(self,
                 vector_retriever,
                 keyword_retriever,
                 hybrid_retriever,
                 generator: Generator,
                 classifier: IntentClassifier,
                 llm,
                 filter_extractor: FilterExtractor = None):

        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.hybrid_retriever = hybrid_retriever
        self.generator = generator
        self.classifier = classifier
        self.llm = llm
        self.filter_extractor = filter_extractor

        self.search_mode = "hybrid"
        self.use_reranker = True
        self.use_filters = True
        self.retrieval_params = {
            "num_final_docs": 5,
            "vector_search_weight": 0.6,
            "hybrid_num_chunks": 5
        }

    def set_search_config(self, search_mode: str = "hybrid", use_reranker: bool = True, use_filters: bool = True):
        if search_mode not in ["vector", "keyword", "hybrid"]:
            raise ValueError("search_mode must be one of: 'vector', 'keyword', 'hybrid'")
        self.search_mode = search_mode
        self.use_reranker = use_reranker
        self.use_filters = use_filters

    def set_retrieval_params(self, **params):
        self.retrieval_params.update(params)

    def _extract_filters_from_query(self, query: str) -> Dict:
        if not self.use_filters or not self.filter_extractor:
            return {}
        try:
            return self.filter_extractor.get_query_filter(query)
        except Exception as e:
            print(f"Error extracting filters: {e}")
            return {}

    def _retrieve_documents_with_filters(self, queries: Union[str, List[str]], filters: Optional[Dict] = None) -> str:
        if isinstance(queries, str):
            queries = [queries]

        if filters is not None:
            try:
                if self.search_mode == "vector":
                    return self._retrieve_vector_with_filters(queries, filters)
                elif self.search_mode == "keyword":
                    return self._retrieve_keyword(queries)
                else:
                    return self._retrieve_hybrid_with_filters(queries, filters)
            except Exception as e:
                print(f"Error in document retrieval: {e}")
                try:
                    return self._retrieve_hybrid_with_filters(queries, {})
                except Exception as fallback_error:
                    print(f"Fallback retrieval also failed: {fallback_error}")
                    return "Không thể truy xuất tài liệu. Vui lòng thử lại sau."
        
        else:
            try:
                if self.search_mode == "vector":
                    query_string = " ".join(queries)
                    extracted_filters = self._extract_filters_from_query(query_string) if self.use_filters else {}
                    return self._retrieve_vector_with_filters(queries, extracted_filters)
                elif self.search_mode == "keyword":
                    return self._retrieve_keyword(queries)
                else:
                    return self._retrieve_hybrid_with_filters(queries, None)
            except Exception as e:
                print(f"Error in document retrieval: {e}")
                try:
                    return self._retrieve_hybrid_with_filters(queries, None)
                except Exception as fallback_error:
                    print(f"Fallback retrieval also failed: {fallback_error}")
                    return "Không thể truy xuất tài liệu. Vui lòng thử lại sau."

    def _retrieve_vector_with_filters(self, queries: List[str], filters: Optional[Dict] = None) -> str:
        chroma_where_clause = None
        if filters and hasattr(self.hybrid_retriever, '_get_filtered_vector_ids'):
            try:
                filtered_vector_ids = self.hybrid_retriever._get_filtered_vector_ids(filters)
                if not filtered_vector_ids:
                    return "Không tìm thấy tài liệu nào phù hợp với bộ lọc đã cung cấp."
                chroma_where_clause = self.hybrid_retriever._create_chroma_where_clause(filtered_vector_ids)
            except Exception as e:
                print(f"Error creating filter for vector search: {e}")

        docs_with_scores = self.vector_retriever.get_unique_parent_docs_with_scores(
            queries, where_clause=chroma_where_clause
        )

        if self.use_reranker and hasattr(self.hybrid_retriever, 'reranker'):
            docs = [doc for doc, _ in docs_with_scores]
            reranked_docs_with_scores = self.hybrid_retriever.reranker.rerank_documents(queries, docs)
            final_docs = [doc for doc, _ in reranked_docs_with_scores[:self.retrieval_params["num_final_docs"]]]
        else:
            final_docs = [doc for doc, _ in docs_with_scores[:self.retrieval_params["num_final_docs"]]]

        return self.hybrid_retriever._format_context(final_docs)

    def _retrieve_keyword(self, queries: List[str]) -> str:

        use_512 = (self.hybrid_retriever.chunk_size == 512)
        docs_with_scores = self.keyword_retriever.get_unique_parent_docs_with_scores(queries, use_512=use_512)
        
        if self.use_reranker and hasattr(self.hybrid_retriever, 'reranker'):
            docs = [doc for doc, _ in docs_with_scores]
            reranked_docs_with_scores = self.hybrid_retriever.reranker.rerank_documents(queries, docs)
            final_docs = [doc for doc, _ in reranked_docs_with_scores[:self.retrieval_params["num_final_docs"]]]
        else:
            final_docs = [doc for doc, _ in docs_with_scores[:self.retrieval_params["num_final_docs"]]]
        
        return self.hybrid_retriever._format_context(final_docs)

    def _retrieve_hybrid_with_filters(self, queries: List[str], filters: Optional[Dict] = None) -> str:
        if hasattr(self.hybrid_retriever, 'retrieve'):
            return self.hybrid_retriever.retrieve(queries, filters=filters)

        try:
            chroma_where_clause = None
            if filters and hasattr(self.hybrid_retriever, '_get_filtered_vector_ids'):
                try:
                    filtered_vector_ids = self.hybrid_retriever._get_filtered_vector_ids(filters)
                    if filtered_vector_ids:
                        chroma_where_clause = self.hybrid_retriever._create_chroma_where_clause(filtered_vector_ids)
                except Exception as e:
                    print(f"Error creating filter for hybrid search: {e}")

            vector_docs_with_scores = self.vector_retriever.get_unique_parent_docs_with_scores(
                queries, where_clause=chroma_where_clause
            )

            use_512 = (self.hybrid_retriever.chunk_size == 512)
            keyword_docs_with_scores = self.keyword_retriever.get_unique_parent_docs_with_scores(queries, use_512=use_512)

            if filters and hasattr(self.hybrid_retriever, '_filter_keyword_results'):
                try:
                    keyword_docs_with_scores = self.hybrid_retriever._filter_keyword_results(keyword_docs_with_scores, filters)
                except Exception as e:
                    print(f"Error filtering keyword results in hybrid search: {e}")

            merged_docs = self.hybrid_retriever._weighted_merge_and_select_docs(
                vector_docs_with_scores,
                keyword_docs_with_scores
            )

            if self.use_reranker and hasattr(self.hybrid_retriever, 'reranker'):
                reranked_docs_with_scores = self.hybrid_retriever.reranker.rerank_documents(queries, merged_docs)
                final_docs = [doc for doc, _ in reranked_docs_with_scores[:self.retrieval_params["num_final_docs"]]]
            else:
                final_docs = merged_docs[:self.retrieval_params["num_final_docs"]]

            return self.hybrid_retriever._format_context(final_docs)
        except Exception as e:
            print(f"Error in manual hybrid retrieval: {e}")
            return "Không thể truy xuất tài liệu. Vui lòng thử lại sau."

    def _retrieve_documents(self, queries: Union[str, List[str]]) -> str:
        return self._retrieve_documents_with_filters(queries)

    def _handle_lookup(self, question_json: Dict, question: str) -> str:
        queries = self.classifier.extract_query(question_json)
        context = self._retrieve_documents_with_filters(queries)
        return self.generator.generate_basic_answer(context, question, self.llm)

    def _handle_comparison(self, question_json: Dict) -> str:
        entities = question_json.get('entities', [])
        topic = question_json.get('topic', '')

        if not entities or len(entities) < 2:
            return self._handle_lookup(question_json, question_json.get('question', ''))

        queries = self.classifier.extract_query(question_json)
        context = self._retrieve_documents_with_filters(queries)
        return self.generator.generate_comparison_answer(context, topic, entities, self.llm)

    def _handle_analysis(self, question_json: Dict, topic_json: Dict) -> str:
        topic = topic_json.get('topic', '')
        queries = self.classifier.extract_query(question_json, topic_json)
        context = self._retrieve_documents_with_filters(queries)
        return self.generator.generate_analysis_answer(context, topic, self.llm)

    def _handle_other(self, question_json: Dict, question: str) -> str:
        return self._handle_lookup(question_json, question)

    def get_search_info(self) -> Dict[str, Any]:
        return {
            "search_mode": self.search_mode,
            "use_reranker": self.use_reranker,
            "use_filters": self.use_filters,
            "retrieval_params": self.retrieval_params.copy(),
            "chunk_size": getattr(self.hybrid_retriever, 'chunk_size', 'unknown'),
            "filter_extractor_available": self.filter_extractor is not None
        }

    def run(self, query: str, explicit_filters: Optional[Dict] = None) -> str:
        try:
            question_json = self.classifier.get_question_json(query)
            topic_json = self.classifier.get_expansion_topic(query)
            intent = question_json.get('intent', 'KHAC')

            if explicit_filters:
                original_use_filters = self.use_filters
                self.use_filters = False
                try:
                    queries = self.classifier.extract_query(question_json)
                    context = self._retrieve_documents_with_filters(queries, explicit_filters)

                    match intent:
                        case 'TRA_CUU':
                            return self.generator.generate_basic_answer(context, query, self.llm)
                        case 'SO_SANH':
                            entities = question_json.get('entities', [])
                            topic = question_json.get('topic', '')
                            if len(entities) < 2:
                                return self.generator.generate_basic_answer(context, query, self.llm)
                            return self.generator.generate_comparison_answer(context, topic, entities, self.llm)
                        case 'PHAN_TICH':
                            topic = topic_json.get('topic', '')
                            return self.generator.generate_analysis_answer(context, topic, self.llm)
                        case _:
                            return self.generator.generate_basic_answer(context, query, self.llm)
                finally:
                    self.use_filters = original_use_filters
            else:
                match intent:
                    case 'TRA_CUU':
                        return self._handle_lookup(question_json, query)
                    case 'SO_SANH':
                        return self._handle_comparison(question_json)
                    case 'PHAN_TICH':
                        return self._handle_analysis(question_json, topic_json)
                    case 'KHAC':
                        return self._handle_lookup(question_json, query)
                    case _:
                        return self._handle_lookup(question_json, query)

        except Exception as e:
            print(f"Error in RAG pipeline: {e}")
            try:
                context = self._retrieve_documents_with_filters([query], explicit_filters)
                return self.generator.generate_basic_answer(context, query, self.llm)
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                return "Xin lỗi, tôi không thể xử lý câu hỏi của bạn lúc này. Vui lòng thử lại sau."