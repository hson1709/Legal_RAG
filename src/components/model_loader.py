from langchain_openai import ChatOpenAI
from config import LLM_MODEL, OPENAI_API_KEY, EMBEDDING_MODEL, DEVICE
from langchain_huggingface import HuggingFaceEmbeddings


def load_embedding_model():

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    return embedding_model

def load_llm():

    llm = ChatOpenAI(
        model = LLM_MODEL,
        openai_api_key = OPENAI_API_KEY,
        temperature=0
    )

    return llm