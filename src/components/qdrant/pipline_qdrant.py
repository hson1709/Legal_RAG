from src.components.qdrant.generator_qdrant import Generator
from src.components.qdrant.bm25_corpus_manager import BM25CorpusManager
from src.components.qdrant.retriever_qdrant import SearchManager

class RAGPipeline:

    def __init__(self,
                 search_manager: SearchManager,
                 generator: Generator,
                 bm25_manager: BM25CorpusManager,
                 llm):

        self.search_manager = search_manager
        self.generator = generator
        self.llm = llm
        self.bm25_manager = bm25_manager

    def run(self, question: str, search_mode: str, search_params: dict = None) -> str:
        """
        Run RAG pipeline with dynamic parameters
        
        Args:
            question: User question
            search_mode: Search mode to use
            search_params: Parameters from UI (optional)
        """
        # Set default params if not provided
        if search_params is None:
            search_params = {}
            
        match search_mode:
            case 'single_vetor':
                # Use parameters from UI or defaults
                vector_name = search_params.get('vector_name', 'content')
                limit = search_params.get('limit', 3)
                
                context = self.search_manager.search_single_vector_context(
                    query=question,
                    vector_name=vector_name,
                    limit=limit
                )
                return self.generator.generate_basic_answer(
                    context=context,
                    question=question,
                    llm=self.llm
                )
            
            case 'multi_vetor':
                # Use parameters from UI or defaults
                vector_names = search_params.get('vector_names', ["content", "title", "summary"])
                limit = search_params.get('limit', 3)
                
                context = self.search_manager.multi_vector_search_context(
                    query=question,
                    vector_names=vector_names,
                    limit=limit
                )
                return self.generator.generate_basic_answer(
                    context=context,
                    question=question,
                    llm=self.llm
                )
            
            case 'weight_single_vetor':
                # Build vector_weights from individual weight parameters
                vector_weights = {
                    "content": search_params.get('content_weight', 0.6),
                    "title": search_params.get('title_weight', 0.3), 
                    "summary": search_params.get('summary_weight', 0.1)
                }
                limit = search_params.get('limit', 5)
                
                context = self.search_manager.weighted_multi_vector_search_context(
                    query=question,
                    vector_weights=vector_weights,
                    limit=limit
                )
                return self.generator.generate_basic_answer(
                    context=context,
                    question=question,
                    llm=self.llm
                )
            
            case 'hybrid':
                # Use parameters from UI
                dense_vectors = search_params.get('dense_vectors', ["content"])
                sparse_weight = search_params.get('sparse_weight', 0.4)
                limit = search_params.get('limit', 5)
                
                # Calculate dense weights
                dense_weight_total = 1.0 - sparse_weight
                dense_weights = {
                    vec: dense_weight_total / len(dense_vectors) 
                    for vec in dense_vectors
                } if dense_vectors else {}
                
                context = self.search_manager.hybrid_search_with_sparse_context(
                    query=question,
                    dense_vectors=dense_vectors,
                    sparse_weight=sparse_weight,
                    dense_weights=dense_weights,
                    limit=limit
                )
                return self.generator.generate_basic_answer(
                    context=context,
                    question=question,
                    llm=self.llm
                )

            case 'rrf_hybrid':
                # Use parameters from UI
                k = search_params.get('k', 60)
                limit = search_params.get('limit', 5)
                
                context = self.search_manager.advanced_hybrid_search_context(
                    query=question,
                    algorithm="rrf",
                    k=k,
                    limit=limit
                )
                return self.generator.generate_basic_answer(
                    context=context,
                    question=question,
                    llm=self.llm
                )

            case 'convex_hybrid':
                # Use parameters from UI
                alpha = search_params.get('alpha', 0.7)
                limit = search_params.get('limit', 5)
                
                context = self.search_manager.advanced_hybrid_search_context(
                    query=question,
                    algorithm="convex",
                    alpha=alpha,
                    limit=limit
                )
                return self.generator.generate_basic_answer(
                    context=context,
                    question=question,
                    llm=self.llm
                )
            
            case _:
                context = self.search_manager.search_single_vector_context(
                    query=question,
                    vector_name="content",
                    limit=5
                )
                return self.generator.generate_basic_answer(
                    context=context,
                    question=question,
                    llm=self.llm
                )