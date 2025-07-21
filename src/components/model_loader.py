from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from config import OPENAI_API_KEY, GOOGLE_API_KEY, EMBEDDING_MODEL, GPT_MODEL, GEMINI_MODEL, DEVICE
from langchain_huggingface import HuggingFaceEmbeddings


def load_embedding_model():

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    return embedding_model


def load_llm(provider="google"):
    if provider == "openai":
        llm = ChatOpenAI(
            model = GPT_MODEL,
            openai_api_key = OPENAI_API_KEY,
            temperature=0
        )
    elif provider == "google":
        llm = ChatGoogleGenerativeAI(
            model = GEMINI_MODEL,
            temperature=0,
            convert_system_message_to_human=True,
            google_api_key = GOOGLE_API_KEY
        )
    else:
        raise ValueError("Unsupported provider")

    return llm
