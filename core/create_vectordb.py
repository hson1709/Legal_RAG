from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import re
from tqdm import tqdm 
import pickle


def clean_metadata(metadata):
    metadata_cleaned = {}
    for key, value in metadata.items():
        if not value:
            metadata_cleaned[key] = value
            continue

        value = str(value).strip()

        if key in {"chuong", "muc"}:
            match = re.search(r"(?:Chương|Mục)?\s*([IVXLCDM\d]+)", value, re.IGNORECASE)
            metadata_cleaned[key] = match.group(1).upper() if match else value

        elif key == "dieu":
            match = re.search(r"(\d+(?:\.\d+)*)(?:\.*)?", value)
            metadata_cleaned[key] = match.group(1) if match else value.lower()

        elif key == "ma_so":
            metadata_cleaned[key] = value

        else:
            metadata_cleaned[key] = value.lower()

    return metadata_cleaned


def create_vectordb(
    docs,
    legal_metadata: list[dict],
    embedding_model,
    persist_directory: str,
    headers_to_split_on=None,
    collection_name="law_docs",
    chunk_size = 512,
    chunk_overlap = 50
):
    if headers_to_split_on is None:
        headers_to_split_on = [
            ("#", "chu_de"),
            ("##", "can_cu"),
            ("###", "chuong"),
            ("####", "muc"),
            ("#####", "dieu")
        ]

    parent_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    parent_documents = []

    for i, raw_doc in enumerate(tqdm(docs, desc="Tạo parent documents")):
        parent_chunks = parent_splitter.split_text(raw_doc)
        base_metadata = legal_metadata[i] if i < len(legal_metadata) else {}

        for chunk in parent_chunks:
            merged_metadata = {
                **chunk.metadata,
                **base_metadata,
                "sourcedoc": f"doc{i}"
            }

            cleaned_metadata = clean_metadata(merged_metadata)

            parent_documents.append(Document(
                page_content=chunk.page_content,
                metadata=cleaned_metadata
            ))

    if not parent_documents:
        raise ValueError("Không có document nào được tạo ra.")

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )

    child_documents = []
    for i, parent_doc in enumerate(tqdm(parent_documents, desc="Tạo child documents")):
        parent_id = f"parent-{i}"
        parent_doc.metadata["parent_id"] = parent_id

        chunks = child_splitter.create_documents(
            [parent_doc.page_content],
            metadatas=[{
                "parent_id": parent_id,
                **parent_doc.metadata
            }]
        )
        child_documents.extend(chunks)

    batch_size = 500
    for i in tqdm(range(0, len(child_documents), batch_size), desc="Lưu vectorstore"):
        vectorstore.add_documents(child_documents[i:i + batch_size])



def create_and_save_parent_documents(
    docs,
    legal_metadata: list[dict],
    persist_path: str,
    headers_to_split_on=None
):
    if headers_to_split_on is None:
        headers_to_split_on = [
            ("#", "chu_de"),
            ("##", "can_cu"),
            ("###", "chuong"),
            ("####", "muc"),
            ("#####", "dieu")
        ]

    parent_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True
    )

    parent_documents = []

    for i, raw_doc in enumerate(tqdm(docs, desc="Tạo parent documents")):
        parent_chunks = parent_splitter.split_text(raw_doc)
        base_metadata = legal_metadata[i] if i < len(legal_metadata) else {}

        for chunk in parent_chunks:
            merged_metadata = {
                **chunk.metadata,
                **base_metadata,
                "sourcedoc": f"doc{i}"
            }

            cleaned_metadata = clean_metadata(merged_metadata)

            parent_documents.append(Document(
                page_content=chunk.page_content,
                metadata=cleaned_metadata
            ))      

    if not parent_documents:
        raise ValueError("Không có document nào được tạo ra.")

    for i, parent_doc in enumerate(parent_documents):
        parent_id = f"parent-{i}"
        parent_doc.metadata["parent_id"] = parent_id  

    with open(persist_path, "wb") as f:
        pickle.dump(parent_documents, f)

    print(f"Đã lưu {len(parent_documents)} parent documents vào {persist_path}")
    return parent_documents


#create_vectordb( 
    docs=markdown_legal_docs,
    legal_metadata=legal_metadata,
    embedding_model=embedding_model,
    persist_directory="./vector_db_512",
    chunk_size = 512,
    chunk_overlap = 50
#)



#create_vectordb( 
    docs=markdown_legal_docs,
    legal_metadata=legal_metadata,
    embedding_model=embedding_model,
    persist_directory="./vector_db_1024",
    chunk_size = 1024,
    chunk_overlap = 100
#)


#parent_docs = create_and_save_parent_documents(
    docs=markdown_legal_docs,
    legal_metadata=legal_metadata,
    persist_path="./parent_documents.pkl"
#)

