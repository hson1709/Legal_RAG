from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from config import PARENT_DOCUMENTS, PERSIST_DIRECTORY
from tqdm import tqdm 
import pickle

class Retriever:
    def __init__(
        self,
        embedding_model,
        pickle_path = PARENT_DOCUMENTS,
        persist_directory = PERSIST_DIRECTORY,
        collection_name="law_docs",
        chunk_size=1024,
        chunk_overlap=100,
        num_chunks=5
    ):
        self.num_chunks = num_chunks
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vectorstore = Chroma(
            embedding_function=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        self.docstore, self.child_documents = self._load_and_process_documents(pickle_path)
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter
        )

    def _load_and_process_documents(self, pickle_path):
        with open(pickle_path, "rb") as f:
            parent_documents = pickle.load(f)
        docstore = InMemoryStore()
        child_documents = []
        store_items = []
        for parent_doc in tqdm(parent_documents, desc="Tạo docstore & child docs"):
            parent_id = parent_doc.metadata["parent_id"]
            store_items.append((parent_id, parent_doc))
            chunks = self.child_splitter.create_documents(
                [parent_doc.page_content],
                metadatas=[{
                    "parent_id": parent_id,
                    **parent_doc.metadata
                }]
            )
            child_documents.extend(chunks)
        docstore.mset(store_items)
        return docstore, child_documents


    def get_context(self, query):
        metadata_label_map = {
            "dieu": "Điều",
            "muc": "Mục",
            "chuong": "Chương",
            "loai_van_ban": "Loại văn bản",
            "chu_de": "Chủ đề",
            "ma_so": "Mã số",
            "ngay_ban_hanh": "Ngày ban hành",
            "noi_ban_hanh": "Nơi ban hành",
            "co_quan_ban_hanh": "Cơ quan ban hành",
            "can_cu": "Căn cứ"
        }
        def format_metadata_vietnamese(metadata):
            parts = []
            for key, value in metadata.items():
                if key in ("sourcedoc", "parent_id"):
                    continue
                label = metadata_label_map.get(key, key)
                parts.append(f"{label}: {value}")
            return "(Metadata: " + "; ".join(parts) + ")"
        child_docs = self.vectorstore.similarity_search(query, k=self.num_chunks)
        unique_parent_docs = {}
        for child_doc in child_docs:
            parent_id = child_doc.metadata.get("parent_id")
            if parent_id and parent_id not in unique_parent_docs:
                parent_doc = self.docstore.mget([parent_id])[0]
                if parent_doc:
                    unique_parent_docs[parent_id] = parent_doc
        parent_docs = list(unique_parent_docs.values())
        contexts = []
        for i, doc in enumerate(parent_docs, 1):
            content = doc.page_content
            metadata_formatted = format_metadata_vietnamese(doc.metadata)
            contexts.append(f"Tài liệu {i}:\n{content}\n{metadata_formatted}")
        context = "\n\n".join(contexts)
        return context
