from utils.prompts import BASIC_PROMT, COMPARISON_PROMPT, ANALYSIS_PROMPT
from langchain_core.prompts import ChatPromptTemplate
from typing import List


class Generator:
    def __init__(self):
        self.prompt_template = self._set_basic_prompt_template()
        self.comparison_template = self._set_comparison_template()
        self.analysis_template = self._set_analysis_template()


    def _set_basic_prompt_template(self):

        system_prompt = BASIC_PROMT
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """
            **Nhiệm vụ**:
                Dưới đây là câu hỏi hoặc yêu cầu bạn cần trả lời:
                {question}

            Tài liệu tham khảo:
            {context}
            """)
        ])
    

    def _set_comparison_template(self):

        system_prompt = COMPARISON_PROMPT
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """
            **Nhiệm vụ**:
                Dựa vào tài liệu tham khảo dưới đây, hãy so sánh điểm giống và khác nhau về {topic} giữa {entities}.

            Tài liệu tham khảo:
            {context}
            """)
        ])
    
    
    def _set_analysis_template(self):

        system_prompt = ANALYSIS_PROMPT
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """
            **Nhiệm vụ**:
            Dựa trên các quy định pháp luật dưới đây, hãy viết một bài phân tích có cấu trúc, logic về {topic}. 

            Tài liệu tham khảo:
            {context}
            """)
        ])
    

    def generate_basic_answer(self, context, question, llm):

        chain = self.prompt_template | llm
        answer = chain.invoke({"question": question, "context": context})
        return answer.content
    
    
    def generate_comparison_answer(self, context: str, topic: str, entities: List[str], llm) -> str:

        entities_str = ' và '.join(entities)
        chain = self.comparison_template | llm
        answer = chain.invoke({
            "topic": topic,
            "entities": entities_str,
            "context": context
        })
        return answer.content
    
    
    def generate_analysis_answer(self, context: str, topic: str, llm) -> str:

        chain = self.analysis_template | llm
        answer = chain.invoke({
            "topic": topic,
            "context": context
        })
        return answer.content
    
    


