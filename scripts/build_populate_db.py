import pickle
import uuid
from typing import List, Dict, Optional
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import pymongo
from pymongo import MongoClient
from src.components.llm_parser import LLMParser
from src.components.model_loader import load_embedding_model
from src.components.data_processor import load_content_doc


def extract_json_from_docs(docs_list: List[str], llm_parser: LLMParser = None) -> List[Dict]:
    return llm_parser.parse(docs_list)


def map_object_to_metadata(obj: Dict) -> Dict:

    hierarchy = obj.get("hierarchy", {})
    
    metadata = {
        "mongo_id": obj.get("_id", ""),
        "ma_so": obj.get("document_code", ""),
        "chu_de": obj.get("document_title", ""),
        "loai_van_ban": obj.get("document_type", ""),
        "co_quan_ban_hanh": obj.get("issuing_authority", ""),
        "noi_ban_hanh": obj.get("issuing_place", ""),
        "ngay_ban_hanh": obj.get("effective_date", ""),
        "tinh_trang": obj.get("status", ""),
        "chuong": hierarchy.get("chapter_number", ""),
        "ten_chuong": hierarchy.get("chapter_title", ""),
        "muc": hierarchy.get("section_number", ""),
        "ten_muc": hierarchy.get("section_title", ""),
        "dieu": hierarchy.get("article_number", ""),
        "ten_dieu": hierarchy.get("article_title", "")
    }
    
    return metadata

def clean_metadata(metadata: Dict) -> Dict:

    cleaned = {}
    for key, value in metadata.items():
        if value is not None and value != "":
            cleaned[key] = str(value)
    return cleaned

def create_vectordb_with_mongodb(
    docs_list: List[str],
    embedding_model,
    vectordb_path: str,
    mongodb_uri: str,
    mongodb_db_name: str,
    mongodb_collection_name: str,
    collection_name: str = "law_docs",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    persist_path: str = "./parent_documents.pkl",
    save_parent_documents: bool = False,
    save_to_mongodb: bool = True,
    json_objects_path: Optional[str] = None
) -> List[Document]:

    
    print("Step 1: Extracting JSON from documents...")
    json_objects = extract_json_from_docs(docs_list)
    
    if not json_objects:
        raise ValueError("Không có object JSON nào được extract thành công.")
    
    print(f"Tổng cộng {len(json_objects)} objects được extract")

    if json_objects_path:
        with open(json_objects_path, "wb") as f:
            pickle.dump(json_objects, f)
        print(f"Đã lưu danh sách JSON object vào {json_objects_path}")
        
    mongo_client = None
    mongo_collection = None
    if save_to_mongodb:
        mongo_client = MongoClient(mongodb_uri)
        mongo_db = mongo_client[mongodb_db_name]
        mongo_collection = mongo_db[mongodb_collection_name]
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=vectordb_path
    )
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    parent_documents = []
    processed_count = 0
    error_count = 0
    
    print("Step 2: Processing objects and creating parent-child documents...")
    
    for i, obj in enumerate(tqdm(json_objects, desc="Processing objects")):
        try:
            content_text = obj.get("content_text", "")
            if not content_text:
                print(f"Object {i}: Không có content_text, bỏ qua")
                error_count += 1
                continue
            
            metadata = map_object_to_metadata(obj)
            cleaned_metadata = clean_metadata(metadata)
            
            parent_id = f"parent-{uuid.uuid4()}"
            cleaned_metadata["parent_id"] = parent_id
            
            parent_doc = Document(
                page_content=content_text,
                metadata=cleaned_metadata
            )
            parent_documents.append(parent_doc)
            
            child_chunks = child_splitter.create_documents(
                [content_text],
                metadatas=[cleaned_metadata]
            )
            
            vector_ids = []
            child_docs_with_ids = []
            
            for chunk in child_chunks:
                vector_id = str(uuid.uuid4())
                vector_ids.append(vector_id)
                chunk.metadata["vector_id"] = vector_id
                child_docs_with_ids.append(chunk)
            
            parent_doc.metadata["vector_id"] = vector_ids
            
            try:
                vectorstore.add_documents(child_docs_with_ids)
            except Exception as e:
                print(f"Object {i}: Lỗi khi lưu vào ChromaDB - {e}")
                error_count += 1
                continue
            
            if save_to_mongodb:
                mongo_doc = {
                    **obj, 
                    "parent_id": parent_id,
                    "vector_id": vector_ids,
                    "metadata": cleaned_metadata
                }
                
                try:
                    mongo_collection.insert_one(mongo_doc)
                except Exception as e:
                    print(f"Object {i}: Lỗi khi lưu vào MongoDB - {e}")
                    error_count += 1
                    continue
            
            processed_count += 1
            if (i + 1) % 100 == 0:
                print(f"Đã xử lý {processed_count} objects thành công, {error_count} lỗi")
                
        except Exception as e:
            print(f"Object {i}: Lỗi không xác định - {e}")
            error_count += 1
            continue
    
    vectorstore.persist()
    
    if save_parent_documents:
        with open(persist_path, "wb") as f:
            pickle.dump(parent_documents, f)
    
    if mongo_client:
        mongo_client.close()
    
    print(f"\n=== KẾT QUẢ XỬ LÝ ===")
    print(f"Tổng objects: {len(json_objects)}")
    print(f"Xử lý thành công: {processed_count}")
    print(f"Lỗi: {error_count}")
    if save_parent_documents:
        print(f"Đã lưu {len(parent_documents)} parent documents vào {persist_path}")
    print(f"Đã lưu vào ChromaDB collection: {collection_name}")
    if save_to_mongodb:
        print(f"Đã lưu vào MongoDB collection: {mongodb_collection_name}")
    else:
        print("Bỏ qua việc lưu vào MongoDB")
    
    return parent_documents




legal_raw_docs = load_content_doc(data_path)
embedding_model = load_embedding_model()
mongodb_config = {
    "mongodb_uri": "mongodb://localhost:27017/",
    "mongodb_db_name": "legal_docs",
    "mongodb_collection_name": "processed_documents"
}

parent_docs = create_vectordb_with_mongodb(
    docs_list=legal_raw_docs,
    embedding_model=embedding_model,
    vectordb_path="./vector_db_512",
    **mongodb_config,
    collection_name="law_docs",
    chunk_size=512,
    chunk_overlap=50,
    save_to_mongodb=True,  
    save_parent_documents=False,
)

parent_docs = create_vectordb_with_mongodb(
    docs_list=legal_raw_docs,
    embedding_model=embedding_model,
    vectordb_path="./vector_db_1024",
    **mongodb_config,
    collection_name="law_docs",
    chunk_size=1024,
    chunk_overlap=100,
    save_to_mongodb=False,
    save_parent_documents=True,
    json_objects_path="./json_docs_objects.pkl"  
)
