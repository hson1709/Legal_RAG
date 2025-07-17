import streamlit as st
from datetime import datetime

from src.pipline import RAGPipeline
from src.components.retriever import Retriever
from src.components.generator import Generator
from src.components.model_loader import load_embedding_model, load_llm

# Cấu hình trang
st.set_page_config(
    page_title="Chatbot Tư vấn Pháp luật",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS để làm đẹp giao diện
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd, #bbdefb);
        border-radius: 10px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        animation: slideIn 0.5s ease-in;
    }
    .user-message {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        color:black
    }
    .bot-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color:black
    }
    .context-info {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px);}
        to { opacity: 1; transform: translateY(0);}
    }
    .stButton > button {
        width: 100%;
        background-color: #1f4e79;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #2c5f99;
    }
    .input-container {
        position: sticky;
        bottom: 0;
        background-color: var(--background-color);
        padding: 1rem 0;
        border-top: 1px solid var(--border-color);
        margin-top: 1rem;
        z-index: 100;
    }
    .stApp {
        background-color: var(--background-color);
    }
    .main .block-container {
        padding-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'pipeline_512' not in st.session_state:
        st.session_state.pipeline_512 = None
    if 'pipeline_1024' not in st.session_state:
        st.session_state.pipeline_1024 = None
    if 'pipeline_loaded' not in st.session_state:
        st.session_state.pipeline_loaded = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False

@st.cache_resource
def load_pipelines():
    embedding_model = load_embedding_model()
    llm = load_llm()
    retriever_512 = Retriever(
        embedding_model=embedding_model,
        persist_directory="./vector_stores/vector_db_512",
        chunk_size=512,
        chunk_overlap=50,
    )
    retriever_1024 = Retriever(
        embedding_model=embedding_model,
        persist_directory="./vector_stores/vector_db_1024",
        chunk_size=1024,
        chunk_overlap=100,
    )
    generator = Generator()
    pipeline_512 = RAGPipeline(retriever_512, generator, llm)
    pipeline_1024 = RAGPipeline(retriever_1024, generator, llm)
    return pipeline_512, pipeline_1024

def display_message(role, content, context=None, timestamp=None):
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑‍💼 Bạn:</strong> {content}
            {f"<br><small>📅 {timestamp}</small>" if timestamp else ""}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Chatbot:</strong><br>{content}
            {f"<br><small>📅 {timestamp}</small>" if timestamp else ""}
        </div>
        """, unsafe_allow_html=True)

def main():
    init_session_state()

    st.markdown('<div class="main-header">⚖️ Chatbot Tư vấn Pháp luật</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Cấu hình")
        chunk_size = st.selectbox(
            "Kích thước chunk:",
            [512, 1024],
            index=1,
            help="Kích thước chunk ảnh hưởng đến độ chính xác và tốc độ xử lý"
        )
        num_chunks = st.number_input(
            "Số lượng tài liệu tham khảo (k):",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Số lượng tài liệu tham khảo sẽ được truy xuất để trả lời"
        )
        st.markdown("---")
        st.info("""
        **Hướng dẫn sử dụng:**
        1. Nhập câu hỏi pháp luật
        2. Chọn kích thước chunk phù hợp
        3. Chọn số lượng tài liệu tham khảo
        4. Nhấn "Gửi" để nhận câu trả lời
        """)
        if st.button("🗑️ Xóa lịch sử trò chuyện"):
            st.session_state.messages = []
            st.success("Đã xóa lịch sử!")
            st.rerun()

    # Load pipelines nếu chưa load
    if not st.session_state.pipeline_loaded:
        with st.spinner(" Đang thiết lập cài đặt..."):
            pipeline_512, pipeline_1024 = load_pipelines()
            st.session_state.pipeline_512 = pipeline_512
            st.session_state.pipeline_1024 = pipeline_1024
            st.session_state.pipeline_loaded = True

        st.success("✅ Các cài đặt được thiết lập thành công!")

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            display_message(
                message["role"],
                message["content"],
                message.get("context"),
                message.get("timestamp")
            )
        if st.session_state.processing:
            with st.spinner("Đang xử lý câu hỏi..."):
                st.empty()

    input_container = st.container()
    with input_container:
        with st.form("question_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                question = st.text_area(
                    "Đặt câu hỏi về văn bản pháp lý:",
                    placeholder="Ví dụ: Thủ tục ly hôn theo pháp luật Việt Nam như thế nào?",
                    help="Nhập câu hỏi về pháp luật Việt Nam",
                    height=90,
                    disabled=st.session_state.processing
                )
            with col2:
                submitted = st.form_submit_button(
                    "📤 Gửi",
                    use_container_width=True,
                    disabled=st.session_state.processing
                )

    if submitted and question.strip() and not st.session_state.processing:
        st.session_state.processing = True
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "timestamp": timestamp
        })
        st.rerun()

    if st.session_state.processing and len(st.session_state.messages) > 0:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "user":
            # Chọn pipeline theo chunk_size
            pipeline = st.session_state.pipeline_1024 if chunk_size == 1024 else st.session_state.pipeline_512
            # Truyền num_chunks vào retriever của pipeline được chọn
            pipeline.retriever.num_chunks = num_chunks
            with st.spinner("Đang tìm kiếm thông tin và tạo câu trả lời..."):
                answer = pipeline.get_answer(last_message["content"])
            if answer:
                answer_timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": answer_timestamp
                })
            st.session_state.processing = False
            st.rerun()
    elif submitted and not question.strip():
        st.warning("⚠️ Vui lòng nhập câu hỏi!")

    with st.container():
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 1rem;">
                <p>🤖 Chatbot Tư vấn Pháp luật - Powered by RAG Technology</p>
                <p><em>Lưu ý: Thông tin chỉ mang tính chất tham khảo, không thay thế tư vấn pháp lý chuyên nghiệp</em></p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()