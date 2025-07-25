from typing import List, Any, Union, Dict
from src.components.generator import Generator
from src.components.intent_classifier import IntentClassifier

class RAGPipeline:
    
    def __init__(self, 
                 vector_retriever, 
                 keyword_retriever, 
                 hybrid_retriever,
                 generator: Generator, 
                 classifier: IntentClassifier, 
                 llm):
   
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.hybrid_retriever = hybrid_retriever
        self.generator = generator
        self.classifier = classifier
        self.llm = llm
        
        self.search_mode = "hybrid"  
        self.use_reranker = True
        self.retrieval_params = {
            "num_final_docs": 5,
            "vector_search_weight": 0.6,
            "hybrid_num_chunks": 5
        }
    
    def set_search_config(self, search_mode: str = "hybrid", use_reranker: bool = True):
 
        if search_mode not in ["vector", "keyword", "hybrid"]:
            raise ValueError("search_mode must be one of: 'vector', 'keyword', 'hybrid'")
            
        self.search_mode = search_mode
        self.use_reranker = use_reranker
    
    def set_retrieval_params(self, **params):

        self.retrieval_params.update(params)
    
    def _retrieve_documents(self, queries: Union[str, List[str]]) -> str:
 
        if isinstance(queries, str):
            queries = [queries]
        
        try:
            if self.search_mode == "vector":
                return self._retrieve_vector(queries)
            elif self.search_mode == "keyword":
                return self._retrieve_keyword(queries)
            else:  # hybrid
                return self._retrieve_hybrid(queries)
                
        except Exception as e:
            print(f"Error in document retrieval: {e}")
            return self._retrieve_hybrid(queries)
    
    def _retrieve_vector(self, queries: List[str]) -> str:
        """Retrieve using vector search"""
        docs_with_scores = self.vector_retriever.get_unique_parent_docs_with_scores(queries)
        
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
    
    def _retrieve_hybrid(self, queries: List[str]) -> str:

        if self.use_reranker:
            return self.hybrid_retriever.retrieve(queries)
        else:
            vector_docs_with_scores = self.vector_retriever.get_unique_parent_docs_with_scores(queries)
            use_512 = (self.hybrid_retriever.chunk_size == 512)
            keyword_docs_with_scores = self.keyword_retriever.get_unique_parent_docs_with_scores(queries, use_512=use_512)
            
            merged_docs = self.hybrid_retriever._weighted_merge_and_select_docs(
                vector_docs_with_scores, 
                keyword_docs_with_scores
            )
            
            final_docs = merged_docs[:self.retrieval_params["num_final_docs"]]
            return self.hybrid_retriever._format_context(final_docs)

    def _handle_lookup(self, question_json: Dict, question: str) -> str:

        queries = self.classifier.extract_query(question_json)
        context = self._retrieve_documents(queries)
        answer = self.generator.generate_basic_answer(context, question, self.llm)
        return answer

    def _handle_comparison(self, question_json: Dict) -> str:

        entities = question_json.get('entities', [])
        topic = question_json.get('topic', '')
        
        if not entities or len(entities) < 2:
            return self._handle_lookup(question_json, question_json.get('question', ''))
        
        queries = self.classifier.extract_query(question_json)
        context = self._retrieve_documents(queries)
        
        answer = self.generator.generate_comparison_answer(
            context, topic, entities, self.llm
        )
        return answer

    def _handle_analysis(self, question_json: Dict, topic_json: Dict) -> str:

        topic = topic_json.get('topic', '')
        queries = self.classifier.extract_query(question_json, topic_json)
        context = self._retrieve_documents(queries)
        
        answer = self.generator.generate_analysis_answer(
            context, topic, self.llm
        )
        return answer
    
    def _handle_other(self, question_json: Dict, question: str) -> str:

        return self._handle_lookup(question_json, question)
    
    def get_search_info(self) -> Dict[str, Any]:

        return {
            "search_mode": self.search_mode,
            "use_reranker": self.use_reranker,
            "retrieval_params": self.retrieval_params.copy(),
            "chunk_size": getattr(self.hybrid_retriever, 'chunk_size', 'unknown')
        }
    
    def run(self, query: str) -> str:

        try:
            question_json = self.classifier.get_question_json(query)
            topic_json = self.classifier.get_expansion_topic(query)
            intent = question_json.get('intent', 'KHAC')

            match intent:
                case 'TRA_CUU':
                    return self._handle_lookup(question_json, query)
                case 'SO_SANH':
                    return self._handle_comparison(question_json)
                case 'PHAN_TICH':
                    return self._handle_analysis(question_json, topic_json)
                case 'KHAC':
                    return self._handle_lookup(question_json)
                
        except Exception as e:
            print(f"Error in RAG pipeline: {e}")
            try:
                context = self._retrieve_documents([query])
                return self.generator.generate_basic_answer(context, query, self.llm)
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                return "Xin lỗi, tôi không thể xử lý câu hỏi của bạn lúc này. Vui lòng thử lại sau."