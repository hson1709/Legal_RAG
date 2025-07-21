from src.components.retriever import Retriever
from src.components.generator import Generator
from src.components.intent_classifier import IntentClassifier

class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator, classifier:IntentClassifier, llm):
        self.retriever = retriever
        self.generator = generator
        self.classifier = classifier
        self.llm = llm

    def _handle_lookup(self, question_json: dict, question) -> str:

        queries = self.classifier.extract_query(question_json)
        context = self.retriever.basic_retrieve(queries)

        answer = self.generator.generate_basic_answer(context, question, self.llm)

        return answer
        

    def _handle_comparison(self, question_json: dict, source_filter = None) -> str:

        entities = question_json['entities']
        topic = question_json['topic']
        
        if not entities or len(entities) < 2:

            return self._handle_lookup(question_json)
        
        queries = self.classifier.extract_query(question_json)

        context = self.retriever.comparison_retrieve(queries, source_filter)
        
        answer = self.generator.generate_comparison_answer(
            context, topic, entities, self.llm
        )

        return answer

    def _handle_analysis(self, question_json, topic_json):

        topic = topic_json['topic']

        queries = self.classifier.extract_query(question_json, topic_json)

        context = self.retriever.analysis_retrieve(queries)

        answer = self.generator.generate_analysis_answer(
            context, topic, self.llm
        )

        return answer
    
    def extract_filter():
        pass
        
    def run(self, query, source_filter = None):

        question_json = self.classifier.get_question_json(query)
        topic_json = self.classifier.get_expansion_topic(query)
        intent = question_json['intent']

        match intent:
            case 'TRA_CUU':
                return self._handle_lookup(question_json, query)
            case 'SO_SANH':
                return self._handle_comparison(question_json, source_filter)
            case 'PHAN_TICH':
                return self._handle_analysis(question_json, topic_json)
            case 'KHAC':
                return self._handle_lookup(question_json)
    



    
