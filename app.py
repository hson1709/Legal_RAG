import streamlit as st
from datetime import datetime

from src.components.model_loader import load_embedding_model, load_llm, load_reranking_model
from src.components.generator import Generator
from src.components.hybrid_retriever_mongo import HybridRetriever
from src.components.vector_retriever import VectorRetriever
from src.components.keyword_retriever import KeywordRetriever
from src.components.intent_classifier import IntentClassifier
from src.components.filter_extractor import FilterExtractor
from src.components.reranker import CrossEncoderReRanker
from src.components.search_mode import SearchMode
from src.pipline import RAGPipeline

st.set_page_config(
    page_title="Chatbot Tư vấn Pháp luật",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .search-mode-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .vector-badge {
        background-color: #e3f2fd;
        color: #1976d2;
        border: 1px solid #bbdefb;
    }
    .keyword-badge {
        background-color: #f3e5f5;
        color: #7b1fa2;
        border: 1px solid #ce93d8;
    }
    .hybrid-badge {
        background-color: #e8f5e8;
        color: #388e3c;
        border: 1px solid #a5d6a7;
    }
    .reranker-badge {
        background-color: #fff3e0;
        color: #f57c00;
        border: 1px solid #ffcc02;
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
    filter_extractor = FilterExtractor(llm=llm)
    reranker = CrossEncoderReRanker(reranking_model)
    
    vector_retriever_512 = VectorRetriever(
        embedding_model=embedding_model,
        persist_directory="./vector_stores/bm_db_legal_512_mongo",
        chunk_size=512,
        chunk_overlap=50,
        num_chunks=10
    )
    
    keyword_retriever_512 = KeywordRetriever(
        corpus_path_512="./data/bm25_corpus_512_mongo.pkl",
        corpus_path_1024="./data/bm25_corpus_1024_mongo.pkl",
        num_chunks=10
    )
    
    hybrid_retriever_512 = HybridRetriever(
        embedding_model=embedding_model,
        persist_directory="./vector_stores/bm_db_legal_512_mongo",
        chunk_size=512,
        chunk_overlap=50,
        reranker=reranker,
        filter_extractor=filter_extractor
    )
    
    vector_retriever_1024 = VectorRetriever(
        embedding_model=embedding_model,
        persist_directory="./vector_stores/bm_db_legal_1024_mongo",
        chunk_size=1024,
        chunk_overlap=100,
        num_chunks=10
    )
    
    keyword_retriever_1024 = KeywordRetriever(
        corpus_path_512="./data/bm25_corpus_512_mongo.pkl",
        corpus_path_1024="./data/bm25_corpus_1024_mongo.pkl",
        num_chunks=10
    )
    
    hybrid_retriever_1024 = HybridRetriever(
        embedding_model=embedding_model,
        persist_directory="./vector_stores/bm_db_legal_1024_mongo",
        chunk_size=1024,
        chunk_overlap=100,
        reranker=reranker,
        filter_extractor=filter_extractor
    )
    
    pipeline_512 = RAGPipeline(
        vector_retriever_512, 
        keyword_retriever_512, 
        hybrid_retriever_512,
        generator, 
        classifier, 
        llm,
        filter_extractor=filter_extractor
    )
    pipeline_1024 = RAGPipeline(
        vector_retriever_1024, 
        keyword_retriever_1024, 
        hybrid_retriever_1024,
        generator, 
        classifier, 
        llm,
        filter_extractor=filter_extractor
    )
    
    return pipeline_512, pipeline_1024

def get_model_info(provider):

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

def get_search_mode_badge(search_mode, use_reranker):

    mode_badges = {
        "vector": '<span class="search-mode-badge vector-badge"> Vector</span>',
        "keyword": '<span class="search-mode-badge keyword-badge"> Keyword</span>',
        "hybrid": '<span class="search-mode-badge hybrid-badge"> Hybrid</span>'
    }
    
    reranker_badge = '<span class="search-mode-badge reranker-badge"> Reranker</span>' if use_reranker else ''
    
    return mode_badges.get(search_mode, '') + reranker_badge

def display_message(role, content, context=None, timestamp=None, model_info=None, search_mode=None, use_reranker=None):
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑‍💼 Bạn:</strong> {content}
            {f"<br><small>📅 {timestamp}</small>" if timestamp else ""}
        </div>
        """, unsafe_allow_html=True)
    else:
        search_badge = get_search_mode_badge(search_mode, use_reranker) if search_mode else ""
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>💡Chatbot:</strong>{search_badge}<br>{content}
            {f"<br><small>📅 {timestamp}</small>" if timestamp else ""}
        </div>
        """, unsafe_allow_html=True)

def update_retriever_params(pipeline, vector_num_chunks, keyword_num_chunks, hybrid_num_chunks, num_final_docs, vector_search_weight):

    pipeline.vector_retriever.num_chunks = vector_num_chunks
    pipeline.keyword_retriever.num_chunks = keyword_num_chunks
    pipeline.hybrid_retriever.vector_retriever.num_chunks = vector_num_chunks
    pipeline.hybrid_retriever.keyword_retriever.num_chunks = keyword_num_chunks
    pipeline.hybrid_retriever.hybrid_num_chunks = hybrid_num_chunks
    pipeline.hybrid_retriever.num_final_docs = num_final_docs
    pipeline.hybrid_retriever.vector_search_weight = vector_search_weight
    
    pipeline.set_retrieval_params(
        num_final_docs=num_final_docs,
        vector_search_weight=vector_search_weight,
        hybrid_num_chunks=hybrid_num_chunks
    )


def main():
    init_session_state()

    st.markdown('<div class="main-header">⚖️ Chatbot Tư vấn Pháp luật</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        st.subheader("Chọn Model AI")
        model_provider = st.selectbox(
            "Mô hình:",
            ["google", "openai"],
            format_func=lambda x: "Google Gemini" if x == "google" else "OpenAI GPT",
            help="Chọn model AI để xử lý câu hỏi của bạn"
        )
        
        st.subheader("🔍 Chế độ tìm kiếm")
        search_mode = st.selectbox(
            "Phương pháp tìm kiếm:",
            ["hybrid", "vector", "keyword"],
            format_func=lambda x: {
                "hybrid": " Hybrid Search (Vector + Keyword)",
                "vector": " Vector Search (Semantic)",
                "keyword": " Keyword Search (BM25)"
            }[x],
            help="Chọn phương pháp tìm kiếm tài liệu"
        )
        
        use_reranker = st.checkbox(
            "📚 Sử dụng Reranker",
            value=True,
            help="Sử dụng mô hình reranking để cải thiện độ chính xác"
        )
        
        st.subheader("📄 Cấu hình Chunk")
        chunk_size = st.selectbox(
            "Kích thước chunk:",
            [512, 1024],
            index=1,
            help="Kích thước chunk ảnh hưởng đến độ chính xác và tốc độ xử lý"
        )
        
        with st.container():
            keyword_num_chunks = 10
            vector_num_chunks = 10
            hybrid_num_chunks = 5
            vector_search_weight = 0.6

            if search_mode in ["vector", "hybrid"]:
                st.markdown("**Vector Retriever:**")
                vector_num_chunks = st.number_input(
                    "Số chunks vector retriever:",
                    min_value=1,
                    max_value=100,
                    value=10,
                    step=1,
                    help="Số lượng chunks được truy xuất bởi vector retriever"
                )

            if search_mode in ["keyword", "hybrid"]:
                st.markdown("**Keyword Retriever:**")
                keyword_num_chunks = st.number_input(
                    "Số chunks keyword retriever:",
                    min_value=1,
                    max_value=100,
                    value=10,
                    step=1,
                    help="Số lượng chunks được truy xuất bởi keyword retriever (BM25)"
                )
            
            if search_mode == "hybrid":
                st.markdown("**Hybrid Parameters:**")
                hybrid_num_chunks = st.number_input(
                    "Số chunks hybrid retriever:",
                    min_value=1,
                    max_value=100,
                        value=5,
                    step=1,
                    help="Số lượng chunks được truy xuất bởi hybrid retriever (dense + sparse)"
                )

            if search_mode == "hybrid":
                vector_search_weight = st.number_input(
                    "Tỉ lệ tài liệu lấy từ vector retriever:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.1,
                    help="Số phần trăm tài liệu lấy từ vector retriever"
                )
            else:
                hybrid_num_chunks = 5
                vector_search_weight = 0.6

            st.markdown("**Final Documents Output:**")
            num_final_docs = st.number_input(
                "Số tài liệu cuối cùng:",
                min_value=1,
                max_value=100,
                value=5,
                step=1,
                help="Số lượng tài liệu cuối cùng được sử dụng để tạo câu trả lời"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🗑️ Xóa lịch sử trò chuyện"):
            st.session_state.messages = []
            st.success("Đã xóa lịch sử!")
            st.rerun()

    if st.session_state.current_model != model_provider or not st.session_state.pipeline_loaded:
        with st.spinner(f"🔄 Đang tải model {get_model_info(model_provider)['name']}..."):
            try:
                pipeline_512, pipeline_1024 = load_pipelines_for_model(model_provider)
                
                modified_pipeline_512 = SearchMode(pipeline_512)
                modified_pipeline_1024 = SearchMode(pipeline_1024)
                
                st.session_state.pipelines[model_provider] = {
                    "512": modified_pipeline_512,
                    "1024": modified_pipeline_1024
                }
                
                st.session_state.current_model = model_provider
                st.session_state.pipeline_loaded = True
                
                st.success(f"✅ Model {get_model_info(model_provider)['name']} đã được tải thành công!")
                
            except Exception as e:
                st.error(f"❌ Lỗi khi tải model: {str(e)}")
                st.stop()

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
                model_info,
                message.get("search_mode"),
                message.get("use_reranker")
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

        st.toast(f"Đang xử lý với {get_model_info(model_provider)['name']}...", icon="⌛")
        st.rerun()

    if st.session_state.processing and len(st.session_state.messages) > 0:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "user":
            try:
                current_pipelines = st.session_state.pipelines.get(model_provider)
                if not current_pipelines:
                    st.error("Pipeline chưa được tải!")
                    st.session_state.processing = False
                    st.stop()
                
                pipeline = current_pipelines["1024"] if chunk_size == 1024 else current_pipelines["512"]
                
                update_retriever_params(
                    pipeline.pipeline, 
                    vector_num_chunks, 
                    keyword_num_chunks, 
                    hybrid_num_chunks,
                    num_final_docs,
                    vector_search_weight
                )
                
                pipeline.set_search_config(search_mode, use_reranker)
                
                with st.spinner(f"Đang tìm kiếm thông tin với {get_model_info(model_provider)['name']}..."):
                    answer = pipeline.run(last_message["content"])
                
                if answer:
                    answer_timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "timestamp": answer_timestamp,
                        "model": model_provider,
                        "search_mode": search_mode,
                        "use_reranker": use_reranker,
                        "params": {
                            "chunk_size": chunk_size,
                            "vector_chunks": vector_num_chunks,
                            "keyword_chunks": keyword_num_chunks,
                            "hybrid_chunks": hybrid_num_chunks,
                            "final_docs": num_final_docs,
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

    with st.container():
        st.markdown("---")
        current_model_info = get_model_info(st.session_state.current_model or model_provider)
        
        search_display = {
            "hybrid": f"Hybrid ({'rerank mode' if use_reranker else 'normal'})",
            "vector": f"Vector ({'rerank mode' if use_reranker else 'normal'})", 
            "keyword": f"Keyword ({'rerank mode' if use_reranker else 'normal'})"
        }[search_mode]
        
        st.markdown(
            f"""
            <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 1rem;">
                <p>🤖 Chatbot Tư vấn Pháp luật - Powered by Hybrid RAG Technology</p>
                <p>Model: <strong>{current_model_info['name']}</strong> | 
                Search: <strong>{search_display}</strong> | 
                Chunk: <strong>{chunk_size}</strong> | 
                Vector: <strong>{vector_num_chunks}</strong> | 
                Keyword: <strong>{keyword_num_chunks}</strong>""" + 
                (f" | Hybrid: <strong>{hybrid_num_chunks}</strong> | Vector Weight: <strong>{vector_search_weight}</strong>" if search_mode == "hybrid" else "") + 
                f""" | Final: <strong>{num_final_docs}</strong></p>
                <p><em>Lưu ý: Thông tin chỉ mang tính chất tham khảo, không thay thế tư vấn pháp lý chuyên nghiệp</em></p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()