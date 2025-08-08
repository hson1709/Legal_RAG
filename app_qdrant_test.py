from src.components.model_loader import load_embedding_model_qdrant, load_llm
from src.components.qdrant.generator_qdrant import Generator
from src.components.qdrant.bm25_corpus_manager import BM25CorpusManager
from src.components.qdrant.retriever_qdrant import SearchManager
from src.components.qdrant.pipline_qdrant import RAGPipeline
from qdrant_client import QdrantClient
from config import QDRANT_API, QDRANT_URL
import pickle


with open("./data/parent_docs_qdrant.pkl", "rb") as f:
    parent_docs = pickle.load(f)


with open("./data/docs_json_format_qdrant.pkl", "rb") as f:
    docs_json_format = pickle.load(f)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API,
)


embedding_model = load_embedding_model_qdrant()
llm = load_llm()
generator = Generator()
bm25_manager = BM25CorpusManager()
bm25_manager.build_corpus_from_documents(docs_json_format)
search_manager = SearchManager(
    embedding_model=embedding_model,
    parent_docs=parent_docs,
    qdrant_client=qdrant_client,
    collection_name="law_docs",
    bm25_manager=bm25_manager
    )
pipline = RAGPipeline(
    search_manager=search_manager,
    generator=generator,
    llm=llm,
    bm25_manager=bm25_manager
)

question = "trình bày kế hoạch thực hiện số hóa sổ hộ tịch trên địa bàn tỉnh tuyên quang"

answer = pipline.run(
    question=question,
    search_mode="hybrid"
)
print(answer)