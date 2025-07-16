from core.rag_pipline import  load_llm, load_embeding_model, get_retriever, create_prompt_template, get_context, get_response
import streamlit as st

import sys
import os
from datetime import datetime


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
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
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
</style>
""", unsafe_allow_html=True)

# Hàm khởi tạo session state
def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = None
    if 'retriever_512' not in st.session_state:
        st.session_state.retriever_512 = None
    if 'retriever_1024' not in st.session_state:
        st.session_state.retriever_1024 = None
    if 'prompt_template' not in st.session_state:
        st.session_state.prompt_template = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False

# Hàm load models một lần
@st.cache_resource
def load_models():
    try:
        with st.spinner("Đang tải các mô hình AI..."):
            # Load embedding model
            embedding_model = load_embeding_model()
            
            # Load retrievers
            retriever_512, _ = get_retriever(
                pickle_path="./data/parent_documents.pkl",
                embedding_model=embedding_model,
                persist_directory="./vector_db_512",
                chunk_size=512,
                chunk_overlap=50
            )
            
            retriever_1024, _ = get_retriever(
                pickle_path="./data/parent_documents.pkl",
                embedding_model=embedding_model,
                persist_directory="./vector_db_1024",
                chunk_size=1024,
                chunk_overlap=100
            )
            
            # Load LLM
            llm = load_llm()
            
            # Create prompt template
            prompt_template = create_prompt_template()
            
            return embedding_model, retriever_512, retriever_1024, llm, prompt_template
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        return None, None, None, None, None

# Hàm xử lý câu hỏi
def process_question(question, chunk_size):
    try:
        # Chọn retriever dựa trên chunk_size
        retriever = st.session_state.retriever_1024 if chunk_size == 1024 else st.session_state.retriever_512
        
        # Lấy context
        with st.spinner("Đang tìm kiếm thông tin liên quan..."):
            context = get_context(question, retriever)
        
        # Tạo câu trả lời
        with st.spinner("Đang tạo câu trả lời..."):
            answer = get_response(
                st.session_state.prompt_template, 
                st.session_state.llm, 
                question, 
                context
            )
        
        return answer, context
    except Exception as e:
        st.error(f"Lỗi khi xử lý câu hỏi: {e}")
        return None, None

# Hàm hiển thị tin nhắn
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
        


# Main app
def main():
    # Khởi tạo session state
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">⚖️ Chatbot Tư vấn Pháp luật</div>', unsafe_allow_html=True)
    
    # Sidebar cho cấu hình
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Tùy chọn chunk size
        chunk_size = st.selectbox(
            "Kích thước chunk:",
            [512, 1024],
            index=1,
            help="Kích thước chunk ảnh hưởng đến độ chính xác và tốc độ xử lý"
        )
        
        st.markdown("---")
        
        # Thông tin về ứng dụng
        st.info("""
        **Hướng dẫn sử dụng:**
        1. Nhập câu hỏi pháp luật
        2. Chọn kích thước chunk phù hợp
        3. Nhấn "Gửi" để nhận câu trả lời
        """)
        
        # Nút xóa lịch sử
        if st.button("🗑️ Xóa lịch sử trò chuyện"):
            st.session_state.messages = []
            st.success("Đã xóa lịch sử!")
            st.rerun()
    
    # Load models nếu chưa load
    if not st.session_state.models_loaded:
        embedding_model, retriever_512, retriever_1024, llm, prompt_template = load_models()
        
        if all([embedding_model, retriever_512, retriever_1024, llm, prompt_template]):
            st.session_state.embedding_model = embedding_model
            st.session_state.retriever_512 = retriever_512
            st.session_state.retriever_1024 = retriever_1024
            st.session_state.llm = llm
            st.session_state.prompt_template = prompt_template
            st.session_state.models_loaded = True
            st.success("✅ Các mô hình đã được tải thành công!")
        else:
            st.error("❌ Không thể tải các mô hình. Vui lòng kiểm tra lại cấu hình.")
            st.stop()
    
    # Phần hiển thị lịch sử trò chuyện
    chat_container = st.container()
    
    with chat_container:
        # Hiển thị tin nhắn từ lịch sử
        for message in st.session_state.messages:
            display_message(
                message["role"], 
                message["content"], 
                message.get("context"), 
                message.get("timestamp")
            )
    
    # Phần nhập câu hỏi
    st.markdown("---")
    
    # Form nhập liệu
    with st.form("question_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            question = st.text_input(
                "Đặt câu hỏi về văn bản pháp lý:",
                placeholder="Ví dụ: Thủ tục ly hôn theo pháp luật Việt Nam như thế nào?",
                help="Nhập câu hỏi về pháp luật Việt Nam"
            )
        
        with col2:
            submitted = st.form_submit_button("📤 Gửi", use_container_width=True)
    
    # Xử lý khi form được submit
    if submitted and question.strip():
        # Thêm câu hỏi vào lịch sử
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "timestamp": timestamp
        })
        
        # Hiển thị câu hỏi mới
        display_message("user", question, timestamp=timestamp)
        
        # Xử lý câu hỏi
        answer, context = process_question(question, chunk_size)
        
        if answer:
            # Thêm câu trả lời vào lịch sử
            answer_timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "context": context,
                "timestamp": answer_timestamp
            })
            
            # Hiển thị câu trả lời
            display_message("assistant", answer, context, answer_timestamp)
            
            # Rerun để cập nhật giao diện
            st.rerun()
    
    elif submitted and not question.strip():
        st.warning("⚠️ Vui lòng nhập câu hỏi!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            <p>🤖 Chatbot Tư vấn Pháp luật - Powered by RAG Technology</p>
            <p><em>Lưu ý: Thông tin chỉ mang tính chất tham khảo, không thay thế tư vấn pháp lý chuyên nghiệp</em></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()