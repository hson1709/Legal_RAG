from src.components.retriever import Retriever
from src.components.generator import Generator

class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator, llm):
        self.retriever = retriever
        self.generator = generator
        self.llm = llm

    def get_answer(self, question):
        
        context = self.retriever.get_context(question)
        answer = self.generator.get_response(self.llm, question, context)
        
        return answer