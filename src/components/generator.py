from utils.prompts import SYSTEM_PROMT
from langchain_core.prompts import ChatPromptTemplate


class Generator:
    def __init__(self):
        self.prompt_template = self._set_prompt_template()

    def _set_prompt_template(self):
        system_prompt = SYSTEM_PROMT
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """
            Dưới đây là câu hỏi bạn cần trả lời:
            {question}

            Tài liệu tham khảo:
            {context}
            """)
        ])

    def get_response(self, llm, question, context):
        chain = self.prompt_template | llm
        answer = chain.invoke({"question": question, "context": context})

        return answer.content

