from src.components.model_loader import load_embedding_model, load_llm, load_reranking_model
from src.components.generator import Generator
from src.components.hybrid_retriever_mongo import HybridRetriever
from src.components.keyword_retriever import KeywordRetriever
from src.components.vector_retriever import VectorRetriever
from src.components.intent_classifier import IntentClassifier
from src.components.filter_extractor import FilterExtractor
from src.components.reranker import CrossEncoderReRanker 
from src.pipline import RAGPipeline


embedding_model = load_embedding_model()
reranking_model = load_reranking_model()
llm = load_llm()
generator = Generator()
classifier = IntentClassifier(llm=llm)
reranker = CrossEncoderReRanker(reranking_model)
filter_extractor = FilterExtractor(llm=llm)

    
# Create retrievers for 512 chunks
vector_retriever_512 = VectorRetriever(
    embedding_model=embedding_model,
    persist_directory="./vector_stores/bm_db_legal_512_mongo",
    chunk_size=512,
    chunk_overlap=50,
    num_chunks=10
)

keyword_retriever_512 = KeywordRetriever(
    corpus_path_512="./data/bm25_corpus_512.pkl",
    corpus_path_1024="./data/bm25_corpus_1024.pkl",
    num_chunks=10
)

hybrid_retriever_512 = HybridRetriever(
    embedding_model=embedding_model,
    persist_directory="./vector_stores/bm_db_legal_512_mongo",
    chunk_size=512,
    chunk_overlap=50,
    reranker=reranker,
    filter_extractor=filter_extractor
)

vector_retriever_1024 = VectorRetriever(
    embedding_model=embedding_model,
    persist_directory="./vector_stores/bm_db_legal_1024_mongo",
    chunk_size=1024,
    chunk_overlap=100,
    num_chunks=10
)

keyword_retriever_1024 = KeywordRetriever(
    corpus_path_512="./data/bm25_corpus_512_mongo.pkl",
    corpus_path_1024="./data/bm25_corpus_1024_mongo.pkl",
    num_chunks=10
)

hybrid_retriever_1024 = HybridRetriever(
    embedding_model=embedding_model,
    persist_directory="./vector_stores/bm_db_legal_1024_mongo",
    chunk_size=1024,
    chunk_overlap=100,
    reranker=reranker,
    filter_extractor=filter_extractor
)

# Create enhanced pipelines
pipeline_512 = RAGPipeline(
    vector_retriever=vector_retriever_512, 
    keyword_retriever=keyword_retriever_512, 
    hybrid_retriever=hybrid_retriever_512,
    generator=generator, 
    classifier=classifier, 
    llm=llm,
    filter_extractor=filter_extractor
)

pipeline_1024 = RAGPipeline(
    vector_retriever=vector_retriever_1024, 
    keyword_retriever=keyword_retriever_1024, 
    hybrid_retriever=hybrid_retriever_1024,
    generator=generator, 
    classifier=classifier, 
    llm=llm,
    filter_extractor=filter_extractor
)

question1 = "so sánh điều 2 nghị quyết 240/NQ-CP và nghị quyết 175/NQ-CP"
question2 = "phân tích chương 1 thông tư 77/2007/TT-BTC"
question3 = "so sánh chương 1 và 2 của thông tư 77/2007/TT-BTC"
question4 = "chương 1 thông tư 77/2007/TT-BTC"
answer = pipeline_1024.run(question3)
print(answer)