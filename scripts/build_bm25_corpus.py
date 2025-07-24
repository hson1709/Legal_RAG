from typing import List
from underthesea import word_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from tqdm import tqdm 
import re
from rank_bm25 import BM25Okapi
from config import PARENT_DOCUMENTS


class BM25CorpusBuilder:
    def __init__(
        self,
        pickle_path = PARENT_DOCUMENTS,
        chunk_size = 1024,
        chunk_overlap = 100
    ):
        self.pickle_path = pickle_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def _load_documents(self):

        with open(self.pickle_path, "rb") as f:
            parent_documents = pickle.load(f)
        return parent_documents

    def _create_child_documents(self, parent_documents):

        child_docs = []
        
        for parent_doc in tqdm(parent_documents, desc="Tạo child documents cho BM25"):
            parent_id = parent_doc.metadata["parent_id"]

            chunks = self.child_splitter.create_documents(
                [parent_doc.page_content],
                metadatas=[{
                    "parent_id": parent_id,
                    "parent_doc": parent_doc,
                    **parent_doc.metadata
                }]
            )
            child_docs.extend(chunks)
        
        return child_docs

    def _preprocess_text(self, text: str) -> List[str]:
   
        text = text.strip()
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        text = ' '.join(text.split())
        
        try:
            tokens = word_tokenize(text, format="text")
        except:
            tokens = text.split()
        
        return tokens

    def _build_bm25_index(self, child_docs):

        print(f"Xây dựng BM25 index cho {len(child_docs)} documents...")
        
        tokenized_corpus = []
        for doc in tqdm(child_docs, desc="Tokenizing documents"):
            tokens = self._preprocess_text(doc.page_content)
            tokenized_corpus.append(tokens)
        
        return tokenized_corpus

    def build_and_save_corpus(self, save_path="bm25_corpus.pkl"):

        print("Bắt đầu xây dựng BM25 corpus...")
        
        parent_documents = self._load_documents()
        
        child_docs = self._create_child_documents(parent_documents)

        print("Xây dựng BM25 index...")
        tokenized_corpus = self._build_bm25_index(child_docs)
  

        corpus_data = {
            'child_docs': child_docs,
            'tokenized_corpus': tokenized_corpus
        }
        
        print(f"Lưu BM25 corpus vào {save_path}...")
        with open(save_path, "wb") as f:
            pickle.dump(corpus_data, f)

corpus_bm25_512 = BM25CorpusBuilder(chunk_size=512, chunk_overlap=50)
corpus_bm25_1024 = BM25CorpusBuilder(chunk_size=1024, chunk_overlap=100)
corpus_bm25_512.build_and_save_corpus(".data/bm25_corpus_512.pkl")
corpus_bm25_1024.build_and_save_corpus(".data/bm25_corpus_1024.pkl")