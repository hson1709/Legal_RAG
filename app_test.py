from src.components.model_loader import load_embedding_model, load_llm, load_reranking_model
from src.components.generator import Generator
from src.components.hybrid_retriever import HybridRetriever
from src.components.intent_classifier import IntentClassifier
from src.components.reranker import CrossEncoderReRanker 
from src.pipline import RAGPipeline


embedding_model = load_embedding_model()
reranking_model = load_reranking_model()
llm = load_llm()
generator = Generator()
classifier = IntentClassifier(llm=llm)
reranker = CrossEncoderReRanker(reranking_model)


retriever_512 = HybridRetriever(
    embedding_model = embedding_model,
    persist_directory = "./vector_stores/bm_db_legal_512",
    chunk_size = 512,
    chunk_overlap = 50,
    reranker=reranker,
    vector_num_chunks=30,
    keyword_num_chunks=30,
    hybrid_num_chunks=10,
    num_final_docs=10
)

retriever_1024 = HybridRetriever(
    embedding_model = embedding_model,
    persist_directory = "./vector_stores/bm_db_legal_1024",
    chunk_size = 1024,
    chunk_overlap = 100,
    reranker=reranker,
    vector_num_chunks=30,
    keyword_num_chunks=30,
    hybrid_num_chunks=20,
    num_final_docs=20
)


pipeline512 = RAGPipeline(
    retriever=retriever_512,
    generator=generator,
    classifier=classifier,
    llm=llm
)

pipeline1024 = RAGPipeline(
    retriever=retriever_1024,
    generator=generator,
    classifier=classifier,
    llm=llm
)

question = "Dựa trên Nghị quyết 1672/NQ-UBTVQH15, hãy phân tích lý do tại sao một số xã mới được đặt tên dựa trên một trong các xã cũ."

answer = pipeline512.run(question)
print(answer)