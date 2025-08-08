import streamlit as st
from datetime import datetime
import pickle
from qdrant_client import QdrantClient

from src.components.model_loader import load_embedding_model_qdrant, load_llm
from src.components.qdrant.generator_qdrant import Generator
from src.components.qdrant.bm25_corpus_manager import BM25CorpusManager
from src.components.qdrant.retriever_qdrant import SearchManager
from src.components.qdrant.pipline_qdrant import RAGPipeline
from config import QDRANT_API, QDRANT_URL

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
    .single-badge {
        background-color: #e1f5fe;
        color: #0277bd;
        border: 1px solid #b3e5fc;
    }
    .multi-badge {
        background-color: #f3e5f5;
        color: #7b1fa2;
        border: 1px solid #ce93d8;
    }
    .weighted-badge {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
    }
    .hybrid-badge {
        background-color: #fff3e0;
        color: #ef6c00;
        border: 1px solid #ffcc02;
    }
    .rrf-badge {
        background-color: #fce4ec;
        color: #c2185b;
        border: 1px solid #f8bbd9;
    }
    .convex-badge {
        background-color: #f1f8e9;
        color: #558b2f;
        border: 1px solid #c5e1a5;
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
def load_pipeline_for_model(model_provider):
    """Load pipeline với Qdrant cho model provider cụ thể"""
    try:
        # Load data
        with open("./data/parent_docs_qdrant.pkl", "rb") as f:
            parent_docs = pickle.load(f)
        
        with open("./data/docs_json_format_qdrant.pkl", "rb") as f:
            docs_json_format = pickle.load(f)
        
        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API,
        )
        
        # Load models với provider cụ thể
        embedding_model = load_embedding_model_qdrant()
        llm = load_llm(provider=model_provider)  # Load với provider
        generator = Generator()
        
        # Initialize BM25
        bm25_manager = BM25CorpusManager()
        bm25_manager.build_corpus_from_documents(docs_json_format)
        
        # Initialize search manager
        search_manager = SearchManager(
            embedding_model=embedding_model,
            parent_docs=parent_docs,
            qdrant_client=qdrant_client,
            collection_name="law_docs",
            bm25_manager=bm25_manager
        )
        
        # Initialize pipeline
        pipeline = RAGPipeline(
            search_manager=search_manager,
            generator=generator,
            llm=llm,
            bm25_manager=bm25_manager
        )
        
        return pipeline, search_manager
        
    except Exception as e:
        st.error(f"Lỗi khi load pipeline cho {model_provider}: {str(e)}")
        return None, None

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
            "description": "Model không xác định",
        }

def get_search_mode_info():
    return {
        "single_vetor": {
            "name": "Single Vector Search",
            "description": "Tìm kiếm sử dụng một vector duy nhất (dense hoặc sparse)",
            "badge_class": "single-badge"
        },
        "multi_vetor": {
            "name": "Multi Vector Search", 
            "description": "Tìm kiếm trên nhiều vectors (content, title, summary)",
            "badge_class": "multi-badge"
        },
        "weight_single_vetor": {
            "name": "Weighted Multi Vector",
            "description": "Tìm kiếm có trọng số trên nhiều vectors",
            "badge_class": "weighted-badge"
        },
        "hybrid": {
            "name": "Hybrid Search",
            "description": "Kết hợp dense (có thể có nhiều dense vectors) và sparse vectors",
            "badge_class": "hybrid-badge"
        },
        "rrf_hybrid": {
            "name": "RRF Hybrid",
            "description": "Hybrid search với Reciprocal Rank Fusion",
            "badge_class": "rrf-badge"
        },
        "convex_hybrid": {
            "name": "Convex Hybrid", 
            "description": "Hybrid search với Convex Combination",
            "badge_class": "convex-badge"
        }
    }

def get_search_mode_badge(search_mode):
    search_modes = get_search_mode_info()
    mode_info = search_modes.get(search_mode, {"badge_class": "single-badge"})
    return f'<span class="search-mode-badge {mode_info["badge_class"]}">{search_modes[search_mode]["name"]}</span>'

def display_message(role, content, timestamp=None, search_mode=None, search_params=None, model_info=None):
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑‍💼 Bạn:</strong> {content}
            {f"<br><small>📅 {timestamp}</small>" if timestamp else ""}
        </div>
        """, unsafe_allow_html=True)
    else:
        search_badge = get_search_mode_badge(search_mode) if search_mode else ""
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>💡 Chatbot:</strong>{search_badge}<br>{content}
            {f"<br><small>📅 {timestamp}</small>" if timestamp else ""}
        </div>
        """, unsafe_allow_html=True)

def render_search_mode_params(search_mode):
    """Render parameters specific to each search mode"""
    params = {}
    
    if search_mode == "single_vetor":
        st.markdown("**Single Vector Parameters:**")
        params['vector_name'] = st.selectbox(
            "Vector để tìm kiếm:",
            ["content", "title", "summary", "sparse_bm25"],
            index=0,
            help="Chọn vector để thực hiện tìm kiếm"
        )
        params['limit'] = st.number_input(
            "Số kết quả:",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Số lượng kết quả trả về"
        )
    
    elif search_mode == "multi_vetor":
        st.markdown("**Multi Vector Parameters:**")
        vector_options = ["content", "title", "summary"]
        params['vector_names'] = st.multiselect(
            "Vectors để tìm kiếm:",
            vector_options,
            default=["content", "title", "summary"],
            help="Chọn các vectors để tìm kiếm"
        )
        params['limit'] = st.number_input(
            "Số kết quả mỗi vector:",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            help="Số lượng kết quả cho mỗi vector"
        )
    
    elif search_mode == "weight_single_vetor":
        st.markdown("**Weighted Multi Vector Parameters:**")
        params['content_weight'] = st.slider(
            "Trọng số Content:",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Trọng số cho vector content"
        )
        params['title_weight'] = st.slider(
            "Trọng số Title:",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Trọng số cho vector title"
        )
        params['summary_weight'] = st.slider(
            "Trọng số Summary:",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Trọng số cho vector summary"
        )
        params['limit'] = st.number_input(
            "Số kết quả cuối cùng:",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Số lượng kết quả cuối cùng"
        )
        
        # Normalize weights
        total_weight = params['content_weight'] + params['title_weight'] + params['summary_weight']
        if total_weight > 0:
            params['content_weight'] /= total_weight
            params['title_weight'] /= total_weight  
            params['summary_weight'] /= total_weight
        else:
            params['content_weight'] = 0.6
            params['title_weight'] = 0.3
            params['summary_weight'] = 0.1
    
    elif search_mode == "hybrid":
        st.markdown("**Hybrid Search Parameters:**")
        dense_options = ["content", "title", "summary"]
        params['dense_vectors'] = st.multiselect(
            "Dense Vectors:",
            dense_options,
            default=["content"],
            help="Chọn dense vectors để kết hợp"
        )
        params['sparse_weight'] = st.slider(
            "Trọng số Sparse Vector:",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1,
            help="Trọng số cho sparse vector (BM25)"
        )
        params['dense_weight'] = 1.0 - params['sparse_weight']
        params['limit'] = st.number_input(
            "Số kết quả cuối cùng:",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Số lượng kết quả cuối cùng"
        )
        st.info(f"Dense weight: {params['dense_weight']:.1f}")
    
    elif search_mode == "rrf_hybrid":
        st.markdown("**RRF Hybrid Parameters:**")
        params['k'] = st.number_input(
            "RRF Parameter K:",
            min_value=1,
            max_value=200,
            value=60,
            step=10,
            help="Parameter k cho Reciprocal Rank Fusion"
        )
        params['limit'] = st.number_input(
            "Số kết quả cuối cùng:",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Số lượng kết quả cuối cùng"
        )
    
    elif search_mode == "convex_hybrid":
        st.markdown("**Convex Hybrid Parameters:**")
        params['alpha'] = st.slider(
            "Alpha (Dense weight):",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Trọng số cho dense vector (1-alpha cho sparse)"
        )
        params['limit'] = st.number_input(
            "Số kết quả cuối cùng:",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Số lượng kết quả cuối cùng"
        )
        st.info(f"Sparse weight: {1-params['alpha']:.1f}")
    
    return params

def update_search_manager_params(search_manager, search_mode, params):
    """Update search manager parameters based on search mode and params"""
    # This function can be extended to modify search_manager properties
    # For now, parameters are passed directly to the search methods
    pass

def main():
    init_session_state()

    st.markdown('<div class="main-header">⚖️ Chatbot Tư vấn Pháp luật - Qdrant Version</div>', unsafe_allow_html=True)

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
        search_modes = get_search_mode_info()
        search_mode = st.selectbox(
            "Phương pháp tìm kiếm:",
            list(search_modes.keys()),
            format_func=lambda x: search_modes[x]["name"],
            help="Chọn phương pháp tìm kiếm tài liệu"
        )
        
        st.info(search_modes[search_mode]["description"])
        
        st.subheader("📊 Tham số tìm kiếm")
        with st.container():
            search_params = render_search_mode_params(search_mode)
        
        if st.button("🗑️ Xóa lịch sử trò chuyện"):
            st.session_state.messages = []
            st.success("Đã xóa lịch sử!")
            st.rerun()

    # Load pipeline if model changed or not loaded
    if st.session_state.current_model != model_provider or not st.session_state.pipeline_loaded:
        with st.spinner(f"🔄 Đang tải model {get_model_info(model_provider)['name']}..."):
            try:
                pipeline, search_manager = load_pipeline_for_model(model_provider)
                if pipeline and search_manager:
                    st.session_state.pipelines[model_provider] = {
                        'pipeline': pipeline,
                        'search_manager': search_manager
                    }
                    st.session_state.current_model = model_provider
                    st.session_state.pipeline_loaded = True
                    st.success(f"✅ Model {get_model_info(model_provider)['name']} đã được tải thành công!")
                else:
                    st.error(f"❌ Không thể tải pipeline cho {get_model_info(model_provider)['name']}!")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Lỗi khi tải model: {str(e)}")
                st.stop()

    # Display messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            model_info = None
            if message["role"] == "assistant" and "model" in message:
                model_info = get_model_info(message["model"])
            
            display_message(
                message["role"],
                message["content"],
                message.get("timestamp"),
                message.get("search_mode"),
                message.get("search_params"),
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

    # Process question
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

    # Generate response
    if st.session_state.processing and len(st.session_state.messages) > 0:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "user":
            try:
                current_pipelines = st.session_state.pipelines.get(model_provider)
                if not current_pipelines:
                    st.error("Pipeline chưa được tải!")
                    st.session_state.processing = False
                    st.stop()
                
                pipeline = current_pipelines['pipeline']
                search_manager = current_pipelines['search_manager']
                
                with st.spinner(f"Đang tìm kiếm thông tin..."):
                    if search_mode == "weight_single_vetor":
                        vector_weights = {
                            "content": search_params['content_weight'],
                            "title": search_params['title_weight'], 
                            "summary": search_params['summary_weight']
                        }
                        context = search_manager.weighted_multi_vector_search_context(
                            query=last_message["content"],
                            vector_weights=vector_weights,
                            limit=search_params['limit']
                        )
                        answer = pipeline.generator.generate_basic_answer(
                            context=context,
                            question=last_message["content"],
                            llm=pipeline.llm
                        )
                    elif search_mode == "hybrid":
                        dense_weights = {vec: search_params['dense_weight'] / len(search_params['dense_vectors']) 
                                    for vec in search_params['dense_vectors']}
                        context = search_manager.hybrid_search_with_sparse_context(
                            query=last_message["content"],
                            dense_vectors=search_params['dense_vectors'],
                            sparse_weight=search_params['sparse_weight'],
                            dense_weights=dense_weights,
                            limit=search_params['limit']
                        )
                        answer = pipeline.generator.generate_basic_answer(
                            context=context,
                            question=last_message["content"],
                            llm=pipeline.llm
                        )
                    elif search_mode == "rrf_hybrid":
                        context = search_manager.advanced_hybrid_search_context(
                            query=last_message["content"],
                            algorithm="rrf",
                            k=search_params['k'],
                            limit=search_params['limit']
                        )
                        answer = pipeline.generator.generate_basic_answer(
                            context=context,
                            question=last_message["content"],
                            llm=pipeline.llm
                        )
                    elif search_mode == "convex_hybrid":
                        context = search_manager.advanced_hybrid_search_context(
                            query=last_message["content"],
                            algorithm="convex",
                            alpha=search_params['alpha'],
                            limit=search_params['limit']
                        )
                        answer = pipeline.generator.generate_basic_answer(
                            context=context,
                            question=last_message["content"],
                            llm=pipeline.llm
                        )
                    else:
                        answer = pipeline.run(
                            question=last_message["content"],
                            search_mode=search_mode,
                            search_params=search_params  
                        )
                
                if answer:
                    answer_timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "timestamp": answer_timestamp,
                        "model": model_provider,
                        "search_mode": search_mode,
                        "search_params": search_params
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
        search_display = search_modes[search_mode]["name"]
        
        # Create params summary
        params_summary = []
        for key, value in search_params.items():
            if isinstance(value, float):
                params_summary.append(f"{key}: {value:.1f}")
            else:
                params_summary.append(f"{key}: {value}")
        
        params_str = " | ".join(params_summary) if params_summary else "Default"
        
        st.markdown(
            f"""
            <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 1rem;">
                <p>🤖 Chatbot Tư vấn Pháp luật - Powered by Qdrant Hybrid RAG</p>
                <p>Model: <strong>{current_model_info['name']}</strong> | Search Mode: <strong>{search_display}</strong> | Parameters: <strong>{params_str}</strong></p>
                <p><em>Lưu ý: Thông tin chỉ mang tính chất tham khảo, không thay thế tư vấn pháp lý chuyên nghiệp</em></p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()