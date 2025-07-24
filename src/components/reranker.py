from typing import List, Any, Union, Tuple

class CrossEncoderReRanker:
    
    def __init__(self, reranking_model):
            self.reranking_model = reranking_model
    
    def rerank_documents(
        self, 
        queries: Union[str, List[str]], 
        documents: List[Any]
    ) -> List[Tuple[Any, float]]:

        if not documents:
            return []

        if isinstance(queries, str):
            queries = [queries]

        doc_scores = {id(doc): 0.0 for doc in documents}

        for query in queries:
            if not query.strip():
                continue

            query_doc_pairs = []
            for doc in documents:
                doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                query_doc_pairs.append([query, doc_text])

            if query_doc_pairs:
                try:
                    scores = self.reranking_model.compute_score(query_doc_pairs)

                    if isinstance(scores, (int, float)):
                        scores = [scores]

                    for doc, score in zip(documents, scores):
                        doc_scores[id(doc)] += float(score)
                        
                except Exception as e:
                    print(f"Warning: Reranking failed for query '{query}': {e}")
                    for doc in documents:
                        doc_scores[id(doc)] += 0.0

        results = [(doc, doc_scores[id(doc)]) for doc in documents]

        results.sort(key=lambda x: x[1], reverse=True)
        
        return results