import streamlit as st
from datetime import datetime

from src.components.model_loader import load_embedding_model, load_llm, load_reranking_model
from src.components.generator import Generator
from src.components.hybrid_retriever import HybridRetriever
from src.components.intent_classifier import IntentClassifier
from src.components.reranker import CrossEncoderReRanker
from src.pipline import RAGPipeline

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
    .model-info {
        background-color: #f0f8ff;
        border-left: 4px solid #007bff;
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .parameter-section {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    .status-indicator {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .status-loaded {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-loading {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'pipelines' not in st.session_state:
        st.session_state.pipelines = {}
    if 'pipeline_loaded' not in st.session_state:
        st.session_state.pipeline_loaded = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None

@st.cache_resource
def load_pipelines_for_model(model_provider):
    """Load both 512 and 1024 pipelines for a model"""
    embedding_model = load_embedding_model()
    llm = load_llm(provider=model_provider)
    generator = Generator()
    classifier = IntentClassifier(llm=llm)
    reranking_model = load_reranking_model()
    reranker = CrossEncoderReRanker(reranking_model)
    
    # Create retriever for 512 chunks
    retriever_512 = HybridRetriever(
        embedding_model=embedding_model,
        persist_directory="./vector_stores/bm_db_legal_512",
        chunk_size=512,
        chunk_overlap=50,
        reranker=reranker
    )
    
    # Create retriever for 1024 chunks
    retriever_1024 = HybridRetriever(
        embedding_model=embedding_model,
        persist_directory="./vector_stores/bm_db_legal_1024",
        chunk_size=1024,
        chunk_overlap=100,
        reranker=reranker
    )
    
    # Create pipelines
    pipeline_512 = RAGPipeline(retriever_512, generator, classifier, llm)
    pipeline_1024 = RAGPipeline(retriever_1024, generator, classifier, llm)
    
    return pipeline_512, pipeline_1024

def get_model_info(provider):
    """Get model information for display"""
    if provider == "openai":
        return {
            "name": "OpenAI GPT",
        }
    elif provider == "google":
        return {
            "name": "Google Gemini",
        }
    else:
        return {
            "name": "Unknown",
            "icon": "❓",
            "description": "Model không xác định",
            "color": "#666666"
        }

def display_message(role, content, context=None, timestamp=None, model_info=None):
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

def update_retriever_params(retriever, vector_num_chunks, keyword_num_chunks, hybrid_num_chunks, num_final_docs, vector_search_weight):
    """Update retriever parameters without reloading"""
    retriever.vector_retriever.num_chunks = vector_num_chunks
    retriever.keyword_retriever.num_chunks = keyword_num_chunks
    retriever.hybrid_num_chunks = hybrid_num_chunks
    retriever.num_final_docs = num_final_docs
    retriever.vector_search_weight = vector_search_weight

def main():
    init_session_state()

    st.markdown('<div class="main-header">⚖️ Chatbot Tư vấn Pháp luật</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Model Selection
        st.subheader("🤖 Chọn Model AI")
        model_provider = st.selectbox(
            "Mô hình:",
            ["google", "openai"],
            format_func=lambda x: "Google Gemini" if x == "google" else "OpenAI GPT",
            help="Chọn model AI để xử lý câu hỏi của bạn"
        )
        
        # Chunk Configuration
        st.subheader("📄 Cấu hình Chunk")
        chunk_size = st.selectbox(
            "Kích thước chunk:",
            [512, 1024],
            index=1,
            help="Kích thước chunk ảnh hưởng đến độ chính xác và tốc độ xử lý"
        )
        
        with st.container():
            
            st.markdown("**Vector Retriever:**")
            vector_num_chunks = st.number_input(
                "Số chunks vector retriever:",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                help="Số lượng chunks được truy xuất bởi vector retriever"
            )
            
            st.markdown("**Keyword Retriever:**")
            keyword_num_chunks = st.number_input(
                "Số chunks keyword retriever:",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                help="Số lượng chunks được truy xuất bởi keyword retriever (BM25)"
            )
            
            st.markdown("**Hybrid with Reranker Documents:**")
            hybrid_num_chunks = st.number_input(
                "Số chunks hybrid retriever:",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="ố lượng chunks được truy xuất bởi hybrid retriever (dense + spare)"
            )

            st.markdown("**Final Documents Output:**")
            num_final_docs = st.number_input(
                "Số tài liệu cuối cùng:",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Số lượng tài liệu cuối cùng sau khi merge từ cả vector và keyword retriever"
            )

            st.markdown("**Vector Search Weight**")
            vector_search_weight = st.number_input(
                "Tỉ lệ tài liệu lấy từ vector retriever:",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Số phần trăm tài liệu lấy từ vector retriever"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        

        # Clear chat history
        if st.button("🗑️ Xóa lịch sử trò chuyện"):
            st.session_state.messages = []
            st.success("Đã xóa lịch sử!")
            st.rerun()


    # Load pipelines when model changes or first load
    if st.session_state.current_model != model_provider or not st.session_state.pipeline_loaded:
        with st.spinner(f"🔄 Đang tải model {get_model_info(model_provider)['name']}..."):
            try:
                pipeline_512, pipeline_1024 = load_pipelines_for_model(model_provider)
                
                # Store pipelines for current model
                st.session_state.pipelines[model_provider] = {
                    "512": pipeline_512,
                    "1024": pipeline_1024
                }
                
                st.session_state.current_model = model_provider
                st.session_state.pipeline_loaded = True
                
                st.success(f"✅ Model {get_model_info(model_provider)['name']} đã được tải thành công!")
                
            except Exception as e:
                st.error(f"❌ Lỗi khi tải model: {str(e)}")
                st.stop()

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            model_info = None
            if message["role"] == "assistant" and "model" in message:
                model_info = get_model_info(message["model"])
            
            display_message(
                message["role"],
                message["content"],
                message.get("context"),
                message.get("timestamp"),
                model_info
            )
        
        if st.session_state.processing:
            with st.spinner("Đang xử lý câu hỏi..."):
                st.empty()

    # Input form
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

    # Handle form submission
    if submitted and question.strip() and not st.session_state.processing:
        st.session_state.processing = True
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "timestamp": timestamp
        })

        st.toast(f"Đang xử lý với {get_model_info(model_provider)['name']}...", icon="⌛")
        st.rerun()

    # Process the question
    if st.session_state.processing and len(st.session_state.messages) > 0:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "user":
            try:
                # Get the appropriate pipeline
                current_pipelines = st.session_state.pipelines.get(model_provider)
                if not current_pipelines:
                    st.error("Pipeline chưa được tải!")
                    st.session_state.processing = False
                    st.stop()
                
                # Select pipeline based on chunk size
                pipeline = current_pipelines["1024"] if chunk_size == 1024 else current_pipelines["512"]
                
                # Update retriever parameters dynamically
                update_retriever_params(
                    pipeline.retriever, 
                    vector_num_chunks, 
                    keyword_num_chunks, 
                    hybrid_num_chunks,
                    num_final_docs,
                    vector_search_weight
                )
                
                with st.spinner(f"Đang tìm kiếm thông tin với {get_model_info(model_provider)['name']}..."):
                    answer = pipeline.run(last_message["content"])
                
                if answer:
                    answer_timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "timestamp": answer_timestamp,
                        "model": model_provider,
                        "params": {
                            "chunk_size": chunk_size,
                            "vector_chunks": vector_num_chunks,
                            "keyword_chunks": keyword_num_chunks,
                            "hybrid_chunks": hybrid_num_chunks,
                            "final_docs" : num_final_docs,
                            "vector_weight": vector_search_weight
                        }
                    })
                else:
                    st.error("Không thể tạo câu trả lời. Vui lòng thử lại!")
                    
            except Exception as e:
                st.error(f"Lỗi khi xử lý: {str(e)}")
            finally:
                st.session_state.processing = False
                st.rerun()
                
    elif submitted and not question.strip():
        st.warning("⚠️ Vui lòng nhập câu hỏi!")

    # Footer
    with st.container():
        st.markdown("---")
        current_model_info = get_model_info(st.session_state.current_model or model_provider)
        st.markdown(
            f"""
            <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 1rem;">
                <p>🤖 Chatbot Tư vấn Pháp luật - Powered by Hybrid RAG Technology</p>
                <p>Đang sử dụng: <strong>{current_model_info['name']}</strong> | 
                Chunk: <strong>{chunk_size}</strong> | 
                Vector: <strong>{vector_num_chunks}</strong> | 
                Keyword: <strong>{keyword_num_chunks}</strong> | 
                Hybrid: <strong>{hybrid_num_chunks}</strong> |
                Final Docs <strong>{num_final_docs}</strong> |
                Vector Weight <strong>{vector_search_weight}</strong></p>
                <p><em>Lưu ý: Thông tin chỉ mang tính chất tham khảo, không thay thế tư vấn pháp lý chuyên nghiệp</em></p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()