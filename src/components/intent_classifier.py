from langchain.schema import HumanMessage
from utils.prompts import INTENT_CLASSIFICATION_PROMPT, EXPANSION_PROMPT
import json, re

class IntentClassifier():
    
    def __init__(self,llm):
        self.llm = llm


    def _clean_json_string(self, raw_content: str) -> str:
        """
        Loại bỏ ```json và ``` nếu xuất hiện trong kết quả của LLM.
        """
        # Loại bỏ ```json và ```
        cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_content.strip())
        return cleaned


    def get_question_json(self, user_query: str) -> dict:
        prompt = INTENT_CLASSIFICATION_PROMPT.replace("{user_query}", user_query)
        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            cleaned_content = self._clean_json_string(response.content)
            question_json = json.loads(cleaned_content)
        except json.JSONDecodeError:
            raise ValueError(f"Lỗi khi parse JSON từ output của LLM:\n{response.content}")

        return question_json


    def get_expansion_topic(self, topic: str) -> dict:
        prompt = EXPANSION_PROMPT.replace("{topic}", topic)
        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            cleaned_content = self._clean_json_string(response.content)
            topic_json = json.loads(cleaned_content)
        except json.JSONDecodeError:
            raise ValueError(f"Lỗi khi parse JSON từ output của LLM:\n{response.content}")

        return topic_json
    

    def extract_query(self, question_json, topic_json = None):
        intent = question_json['intent']
        topic = question_json['topic']
        entities = question_json['entities']
        sub_topics = topic_json['sub_topic'] if topic_json else []
        query = []

        def get_sub_query(topic, entities):
            if not entities:
                query.append(topic)
            else:
                for entity in entities:
                    sub_query = f"{topic} {entity}"
                    query.append(sub_query)


        def get_expansion_sub_query(sub_topics, entities):
            if not sub_topics:
                if not entities:
                    query.append(topic)
                else:
                    for entity in entities:
                        sub_query = f"{topic} {entity}"
                        query.append(sub_query)
            else:
                if not entities:
                    for sub_topic in sub_topics:
                        query.append(sub_topic)
                else:
                    for entity in entities:
                        for sub_topic in sub_topics:
                            sub_query = f"{sub_topic} {entity}"
                            query.append(sub_query)


        match intent:
            case 'TRA_CUU' | 'SO_SANH' | 'KHAC':
                get_sub_query(topic, entities)
            case 'PHAN_TICH':
                get_expansion_sub_query(sub_topics, entities)

        return query



