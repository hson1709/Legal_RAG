from utils.prompts import BASIC_PROMT
from langchain_core.prompts import ChatPromptTemplate


class Generator:
    def __init__(self):
        self.prompt_template = self._set_basic_prompt_template()


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
    

    def generate_basic_answer(self, context, question, llm):

        chain = self.prompt_template | llm
        answer = chain.invoke({"question": question, "context": context})
        return answer.content