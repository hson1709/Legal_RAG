from src.pipline import RAGPipeline

class SearchMode:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.search_mode = "hybrid"
        self.use_reranker = True

    def set_search_config(self, search_mode, use_reranker):
        self.search_mode = search_mode
        self.use_reranker = use_reranker
        self.pipeline.set_search_config(search_mode, use_reranker)

    def run(self, query):
        return self.pipeline.run(query)
    
    @property
    def retriever(self):
        return self.pipeline