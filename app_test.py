from src.components.model_loader import load_embedding_model, load_llm
from src.components.generator import Generator
from src.components.retriever import Retriever
from src.components.intent_classifier import IntentClassifier
from src.pipline import RAGPipeline


embedding_model = load_embedding_model()
llm = load_llm()
generator = Generator()
classifier = IntentClassifier(llm=llm)

retriever_512 = Retriever(
    embedding_model=embedding_model,
    persist_directory="./vector_stores/bm_db_legal_512",
    chunk_size=512,
    chunk_overlap=50,
)

retriever_1024 = Retriever(
    embedding_model=embedding_model,
    persist_directory="./vector_stores/bm_db_legal_1024",
    chunk_size=1024,
    chunk_overlap=100,
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

question = "So sánh số lượng xã được giữ nguyên không sắp xếp tại tỉnh Lâm Đồng (theo Nghị quyết 1671/NQ-UBTVQH15) với số lượng đơn vị hành chính cấp xã tại tỉnh Lạng Sơn trước khi sắp xếp (theo Nghị quyết 1672/NQ-UBTVQH15)"

answer = pipeline512.run(question)
print(answer)