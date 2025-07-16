from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from utils.config import LLM_MODEL, OPENAI_API_KEY, EMBEDDING_MODEL, DEVICE
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm 
import pickle


def load_embeding_model():
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



def get_retriever(
    pickle_path: str,
    embedding_model,
    persist_directory: str,
    collection_name="law_docs",
    chunk_size=1024,
    chunk_overlap=100
):
    
    with open(pickle_path, "rb") as f:
        parent_documents = pickle.load(f)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )

    docstore = InMemoryStore()
    store_items = []

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    child_documents = []

    for _, parent_doc in enumerate(tqdm(parent_documents, desc="Tạo docstore & child docs")):
        parent_id = parent_doc.metadata["parent_id"]
        store_items.append((parent_id, parent_doc))

        chunks = child_splitter.create_documents(
            [parent_doc.page_content],
            metadatas=[{
                "parent_id": parent_id,
                **parent_doc.metadata
            }]
        )
        child_documents.extend(chunks)

    docstore.mset(store_items)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter
    )

    return retriever, child_documents


def create_prompt_template():
  
  system_prompt = """Bạn là một chuyên gia tư vấn pháp luật với 30 năm kinh nghiệm trong lĩnh vực này.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng về pháp luật của Việt Nam dựa trên thông tin được cung cấp.
Bạn phải đảm bảo trả lời câu hỏi một các chính xác, đầy đủ, dễ hiểu và phù hợp với các quy định pháp luật hiện hành.

**Yêu cầu trả lời:**

### 1. Phân tích câu hỏi:
- Hiểu rõ toàn diện về câu hỏi, bao gồm các cách diễn đạt tương tự, các từ đồng nghĩa và các biên thể ngữ nghĩa hoặc cách diễn đạt không trực tiếp.
- Xác định rõ các khía cạnh chính hoặc các điểm cần làm rõ từ câu hỏi

### 2. Cấu trúc câu trả lời:
- Mở đầu bằng một câu trả lời chính cho câu hỏi một cách tổng hợp tóm tắt, ngắn gọn, bao quát đầy đủ ý, đi thẳng vào trọng tâm vấn đề chính.
- Trả lời theo từng ý rõ ràng, mỗi ý tương ứng với một điểm hoặc khía cạnh cụ thể.
- Sử dụng các số thứ tự (1, 2, 3...) hoặc dấu đầu dòng để trình bày các ý của nội dung một cách logic .
- Lưu ý bắt buộc: Với **mỗi ý** trong câu trả lời, bạn **bắt buộc phải trích dẫn chính xác nguồn từ phần Metadata** của tài liệu tham khảo tương ứng khi dùng để trả lời ý đó.

### 3. **Trích dẫn nguồn:** (ĐẶC BIỆT QUAN TRỌNG)
- Trích dẫn nuồn theo mẫu sau: (*Nguồn: Điều [Điều] - Mục [Mục] - Chương [Chương], văn bản [Loại văn bản]: [Chủ đề], số [Mã số], ban hành ngày [Ngày ban hành], tại [Nơi ban hành], bởi [Cơ quan ban hành]).
- Các ví dụ mẫu về cách chuyển từ Metadata gốc sang format trích dẫn nguồn như sau:
1. (Metadata: Điều: Điều 14; Mục: Mục 3; Chương: Chương II; Loại văn bản: nghị quyết; Chủ đề: phiên họp chuyên đề về xây dựng pháp luật tháng 8 năm 2023 chính phủ; Mã số: 64/2025/QH15; Ngày ban hành: 19/02/2025; Nơi ban hành: hà nội; Cơ quan ban hành: quốc hội)
(*Nguồn: Điều 14 - Mục 3 - Chương II, văn bản Nghị quyết: Phiên họp chuyên đề về xây dựng pháp luật tháng 8 năm 2023 chính phủ, số 64/2025/QH15, ban hành ngày 19/02/2025 tại Hà Nội, bởi Quốc hội)

2. (Metadata: Điều: Điều 47; Loại văn bản: luật; Chủ đề: ban hành văn bản quy phạm pháp luật; Mã số: 64/2025/QH15; Ngày ban hành: 19/02/2025; Nơi ban hành: hà nội; Cơ quan ban hành: quốc hội)
(*Nguồn: Điều 47, văn bản Luật: Ban hành văn bản quy phạm pháp luật, số 64/2025/QH15, ban hành ngày 19/02/2025 tại Hà Nội, bởi Quốc hội)

3. (Metadata: Điều: Điều 4; Chương: Chương I; Loại văn bản: nghị định; Chủ đề: quy định chi tiết một số điều của luật an toàn thực phẩm; Mã số: 15/2022/NĐ-CP; Ngày ban hành: 28/01/2022; Nơi ban hành: hà nội; Cơ quan ban hành: chính phủ)
(*Nguồn: Điều 4 - Chương I, văn bản Nghị định: Quy định chi tiết một số điều của Luật An toàn thực phẩm, số 15/2022/NĐ-CP, ban hành ngày 28/01/2022, tại Hà Nội, bởi Chính phủ)

4. (Metadata: Điều: Điều 9; Mục: Mục 5; Chương: Chương I; Loại văn bản: thông tư; Chủ đề: hướng dẫn kỹ thuật về phòng ngừa, ứng phó sự cố chất thải; Mã số: 2025/TT-BNNMT; Ngày ban hành: 14/07/2025; Nơi ban hành: hà nội; Cơ quan ban hành: bộ nông nghiệp và môi trường)
(*Nguồn: Điều 9 - Mục 5 - Chương I, văn bản Thông tư: Hướng dẫn kỹ thuật về phòng ngừa, ứng phó sự cố chất thải, số 2025/TT-BNNMT, ban hành ngày 14/07/2025, tại Hà Nội, bởi Bộ Nông nghiệp và môi trường)

5. (Metadata: Điều: Điều 1; Mục: Mục 4; Loại văn bản: kế hoạch; Chủ đề: xây dựng chính sách năm 2014; Mã số: 32/KH/CP; Ngày ban hành: 21/06/2024; Nơi ban hành: hà nội; Cơ quan ban hành: chính phủ)
(*Nguồn: Điều 1, Mục 4, văn bản Kế hoạch: Xây dựng chính sách năm 2014, số 32/KH/CP, ban hành ngày 21/06/2024 tại Hà Nội, bởi Chính phủ)

- Bắt buộc phải trích dẫn theo đúng định dạng trong mẫu.
- Phải tìm và trích dẫn nguồn từ phần Metadata sử dụng để trả lời với mỗi ý.
- Metadata được cung cấp ở cuối mỗi tài liệu, vì vậy sẽ luôn có nguồn.

### 4. Tuân thủ nội dung:
- Chỉ sử dụng thông tin từ nội dung được cung cấp trong phần 'Tài liệu tham khảo:' để trả lời câu hỏi, có thể suy diễn và suy luận từ thông tin. Nhưng tuyệt đối không thêm thông tin từ bên ngoài.

### 5. Xử lý trường hợp không có thông tin:
- Nêu không có thông tin phù hợp trong dữ liệu cung cấp, bắt buộc phải trả lời như sau: 'Tôi không tìm thấy thông tin trong tài liệu'.

### 6. Phong cách trình bày:
- Không quá dài, rõ ràng nhưng vẫn đảm bảo đủ độ chi tiết, đủ ý và thông tin.
- Chuyên nghiệp và chính xác theo ngôn ngữ pháp luật Việt Nam.
- Tránh sử dụng ngôn ngữ không trang trọng.

### 7. Đảm bảo chất lượng:
- Mọi câu trả lời cần được kiểm tra để đảm bảo tối đa tính chính xác và rõ ràng trước khi gửi đi.

Lưu ý quan trọng: Tất cả các những yêu cầu trên chỉ dùng để hướng dẫn những quy định khi bạn trả lời câu hỏi, khi trả lời bạn chỉ cần đảm bảo các yêu cầu trên, bắt buộc khi trả lời chỉ thực hiện trả lời theo yêu cầu và tuyệt đối không được tạo ra các yêu cầu ### phía trên.
"""

  prompt_template = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      ("human", """
      Dưới đây là câu hỏi bạn cần trả lời:
      {question}

      Tài liệu tham khảo:
      {context}
      """)
  ])

  return prompt_template


def get_context(query, retriever):

    metadata_label_map = {
    "dieu": "Điều",
    "muc" : "Mục",
    "chuong": "Chương",
    "loai_van_ban": "Loại văn bản",
    "chu_de": "Chủ đề",
    "ma_so": "Mã số",
    "ngay_ban_hanh": "Ngày ban hành",
    "noi_ban_hanh": "Nơi ban hành",
    "co_quan_ban_hanh": "Cơ quan ban hành",
    "can_cu" : "Căn cứ"
    }

    def format_metadata_vietnamese(metadata):
        parts = []
        for key, value in metadata.items():
            if key in ("sourcedoc", "parent_id"):
                label = metadata_label_map.get(key, key)
                continue

            label = metadata_label_map.get(key, key)
            parts.append(f"{label}: {value}")

        return "(Metadata: " + "; ".join(parts) + ")"


    child_docs = retriever.vectorstore.similarity_search(query, k=5)

    unique_parent_docs = {}
    for child_doc in child_docs:
        parent_id = child_doc.metadata.get("parent_id")
        if parent_id and parent_id not in unique_parent_docs:
            parent_doc = retriever.docstore.mget([parent_id])[0]
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


def get_response(prompt_template, llm, question, context):
    chain = prompt_template | llm
    answer = chain.invoke({"question": question, "context": context})

    return answer.content




