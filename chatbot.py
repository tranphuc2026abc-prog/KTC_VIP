import os
import glob
import base64
import streamlit as st
import shutil
from pathlib import Path

# --- Imports với xử lý lỗi ---
try:
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from groq import Groq
    # [NÂNG CẤP] Thêm thư viện Rerank
    from flashrank import Ranker, RerankRequest
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG (CONFIG)
# ==============================================================================

st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="LOGO.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    # Model Config
    LLM_MODEL = 'llama-3.1-8b-instant'
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # [NÂNG CẤP] Model Rerank nhỏ nhẹ, chạy tốt trên CPU
    RERANK_MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"
    
    # Paths
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    RERANK_CACHE = "./opt" # Nơi lưu cache model rerank
    
    # Assets
    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"
    
    # RAG Parameters
    CHUNK_SIZE = 1000       # Giảm nhẹ size để đoạn văn tập trung hơn
    CHUNK_OVERLAP = 200     
    RETRIEVAL_K = 20        # [NÂNG CẤP] Lấy 20 đoạn thô ban đầu (thay vì 6)
    FINAL_K = 5             # Chỉ lấy 5 đoạn tốt nhất sau khi Rerank để gửi cho AI
    RETRIEVAL_TYPE = "mmr" 

# ==============================================================================
# 2. XỬ LÝ GIAO DIỆN (UI MANAGER)
# ==============================================================================

class UIManager:
    @staticmethod
    def get_img_as_base64(file_path):
        if not os.path.exists(file_path):
            return ""
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    @staticmethod
    def inject_custom_css():
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
            html, body, [class*="css"], .stMarkdown, .stButton, .stTextInput, .stChatInput {
                font-family: 'Inter', sans-serif !important;
            }
            section[data-testid="stSidebar"] {
                background-color: #f8f9fa; border-right: 1px solid #e9ecef;
            }
            .project-card {
                background: white; padding: 15px; border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px;
                border: 1px solid #dee2e6;
            }
            .project-title {
                color: #0077b6; font-weight: 800; font-size: 1.1rem;
                margin-bottom: 5px; text-align: center; text-transform: uppercase;
            }
            .project-sub {
                font-size: 0.8rem; color: #6c757d; text-align: center;
                margin-bottom: 15px; font-style: italic;
            }
            .main-header {
                background: linear-gradient(135deg, #023e8a 0%, #0077b6 100%);
                padding: 1.5rem 2rem; border-radius: 15px; color: white;
                margin-bottom: 2rem; box-shadow: 0 8px 20px rgba(0, 119, 182, 0.3);
                display: flex; align-items: center; justify-content: space-between;
            }
            .header-left h1 {
                color: #caf0f8 !important; font-weight: 900; margin: 0;
                font-size: 2.2rem; letter-spacing: -0.5px;
            }
            .header-left p {
                color: #e0fbfc; margin: 5px 0 0 0; font-size: 1rem; opacity: 0.9;
            }
            .header-right img {
                border-radius: 50%; border: 3px solid rgba(255,255,255,0.3);
                box-shadow: 0 4px 10px rgba(0,0,0,0.2); width: 100px; height: 100px;
                object-fit: cover;
            }
            [data-testid="stChatMessageContent"] {
                border-radius: 15px !important; padding: 1rem !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            [data-testid="stChatMessageContent"]:has(+ [data-testid="stChatMessageAvatar"]) {
                background: #e3f2fd; color: #0d47a1;
            }
            [data-testid="stChatMessageContent"]:not(:has(+ [data-testid="stChatMessageAvatar"])) {
                background: white; border: 1px solid #e9ecef;
                border-left: 5px solid #00b4d8;
            }
            div.stButton > button {
                border-radius: 8px; background-color: white; color: #0077b6;
                border: 1px solid #90e0ef; transition: all 0.2s;
            }
            div.stButton > button:hover {
                background-color: #0077b6; color: white;
                border-color: #0077b6; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_sidebar():
        with st.sidebar:
            if os.path.exists(AppConfig.LOGO_SCHOOL):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(AppConfig.LOGO_SCHOOL, use_container_width=True)
                st.markdown("<div style='text-align:center; font-weight:700; color:#023e8a; margin-bottom:20px;'>THCS & THPT PHẠM KIỆT</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="project-card">
                <div class="project-title">KTC CHATBOT</div>
                <div class="project-sub">Sản phẩm dự thi KHKT cấp trường</div>
                <hr style="margin: 10px 0; border-top: 1px dashed #dee2e6;">
                <div style="font-size: 0.9rem; line-height: 1.6;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: 600; color: #555;">Tác giả:</span>
                        <span style="text-align: right; color: #222;"><b>Bùi Tá Tùng</b><br><b>Cao Sỹ Bảo Chung</b></span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                        <span style="font-weight: 600; color: #555;">GVHD:</span>
                        <span style="text-align: right; color: #222;">Thầy <b>Nguyễn Thế Khanh</b></span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                        <span style="font-weight: 600; color: #555;">Năm học:</span>
                        <span style="text-align: right; color: #222;"><b>2025 - 2026</b></span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### ⚙️ Tiện ích")
            if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            
            # Nút Rebuild DB
            if st.button("🔄 Cập nhật dữ liệu mới", use_container_width=True):
                if os.path.exists(AppConfig.VECTOR_DB_PATH):
                    shutil.rmtree(AppConfig.VECTOR_DB_PATH)
                st.session_state.pop('vector_db', None)
                st.rerun()

    @staticmethod
    def render_header():
        logo_nhom_b64 = UIManager.get_img_as_base64(AppConfig.LOGO_PROJECT)
        img_html = f'<img src="data:image/jpeg;base64,{logo_nhom_b64}" alt="Logo">' if logo_nhom_b64 else ""

        st.markdown(f"""
        <div class="main-header">
            <div class="header-left">
                <h1>KTC CHATBOT</h1>
                <p style="font-size: 1.1rem; margin-top: 5px;">Học Tin dễ dàng - Thao tác vững vàng</p>
            </div>
            <div class="header-right">
                {img_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 3. LOGIC BACKEND (RAG ENGINE + RERANK)
# ==============================================================================

class RAGEngine:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_groq_client():
        try:
            api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
            if not api_key: return None
            return Groq(api_key=api_key)
        except: return None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_embedding_model():
        try:
            return HuggingFaceEmbeddings(
                model_name=AppConfig.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            st.error(f"Lỗi tải Embedding Model: {e}")
            return None

    # [NÂNG CẤP] Load Model Rerank (Chỉ load 1 lần)
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_reranker():
        try:
            return Ranker(model_name=AppConfig.RERANK_MODEL_NAME, cache_dir=AppConfig.RERANK_CACHE)
        except Exception as e:
            print(f"Lỗi tải Reranker: {e}")
            return None

    @staticmethod
    def build_or_load_vector_db(embeddings):
        if not embeddings: return None

        # 1. Thử load từ ổ cứng
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                return FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                st.warning(f"Không thể tải dữ liệu cũ: {e}. Đang tạo mới...")

        # 2. Tạo mới từ PDF
        if not os.path.exists(AppConfig.PDF_DIR):
            st.error(f"⚠️ Thư mục '{AppConfig.PDF_DIR}' không tồn tại!")
            return None

        pdf_files = glob.glob(os.path.join(AppConfig.PDF_DIR, "*.pdf"))
        txt_files = glob.glob(os.path.join(AppConfig.PDF_DIR, "*.txt"))
        all_files = pdf_files + txt_files
        
        if not all_files:
            st.warning("⚠️ Không tìm thấy tài liệu PDF/TXT nào!")
            return None

        docs = []
        status_text = st.empty()
        status_text.info(f"📚 Đang số hóa {len(all_files)} tài liệu. Vui lòng đợi...")

        for file_path in all_files:
            try:
                source_name = os.path.basename(file_path).replace('.pdf', '').replace('.txt', '').replace('_', ' ')
                
                content = ""
                if file_path.endswith('.pdf'):
                    reader = PdfReader(file_path)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text and len(text.strip()) > 50:
                            clean_text = text.replace('\x00', '')
                            # [NÂNG CẤP] Contextual Embedding: Dán nhãn tên sách vào nội dung
                            context_content = f"Tài liệu môn: {source_name}\nNội dung chi tiết: {clean_text}"
                            
                            docs.append(Document(
                                page_content=context_content, 
                                metadata={"source": os.path.basename(file_path), "page": page_num + 1}
                            ))
                elif file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        if text:
                            context_content = f"Tài liệu môn: {source_name}\nNội dung chi tiết: {text}"
                            docs.append(Document(
                                page_content=context_content,
                                metadata={"source": os.path.basename(file_path), "page": 1}
                            ))
            except Exception as e:
                print(f"Lỗi đọc file {file_path}: {e}")
                continue

        if docs:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=AppConfig.CHUNK_SIZE, 
                chunk_overlap=AppConfig.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            splits = splitter.split_documents(docs)
            vector_db = FAISS.from_documents(splits, embeddings)
            vector_db.save_local(AppConfig.VECTOR_DB_PATH)
            status_text.empty()
            return vector_db
        
        status_text.empty()
        return None

    @staticmethod
    def generate_response(client, vector_db, query):
        context_text = ""
        sources = []
        
        if vector_db:
            # [NÂNG CẤP] Bước 1: Lấy diện rộng (20 đoạn)
            retriever = vector_db.as_retriever(
                search_type=AppConfig.RETRIEVAL_TYPE, 
                search_kwargs={"k": AppConfig.RETRIEVAL_K, "fetch_k": 50}
            )
            initial_docs = retriever.invoke(query)
            
            # [NÂNG CẤP] Bước 2: Rerank (Lọc tinh hoa)
            ranker = RAGEngine.load_reranker()
            final_docs = []

            if ranker and initial_docs:
                # Chuẩn bị dữ liệu cho FlashRank
                passages = [
                    {"id": str(i), "text": doc.page_content, "meta": doc.metadata} 
                    for i, doc in enumerate(initial_docs)
                ]
                
                # Thực hiện chấm điểm lại
                rerank_request = RerankRequest(query=query, passages=passages)
                ranked_results = ranker.rank(rerank_request)
                
                # Chỉ lấy Top 5 kết quả tốt nhất
                top_results = ranked_results[:AppConfig.FINAL_K]
                
                # Tạo lại danh sách docs
                for res in top_results:
                    final_docs.append(Document(page_content=res['text'], metadata=res['meta']))
            else:
                # Fallback nếu Reranker lỗi
                final_docs = initial_docs[:AppConfig.FINAL_K]

            # Tạo ngữ cảnh prompt
            for doc in final_docs:
                src = doc.metadata.get('source', 'Tài liệu')
                page = doc.metadata.get('page', 'Unknown')
                content = doc.page_content.replace("\n", " ").strip()
                
                context_text += f"""
                ---
                [Tài liệu: {src}, Trang: {page}]
                {content}
                ---
                """
                sources.append(f"{src} - Trang {page}")

        system_prompt = f"""Bạn là KTC Chatbot - Trợ lý AI giáo dục của trường Phạm Kiệt.
        
        NHIỆM VỤ:
        1. Trả lời câu hỏi dựa CHÍNH XÁC vào [NGỮ CẢNH] bên dưới.
        2. Chú ý phân biệt tài liệu của các khối lớp (Tin 10, Tin 11, Tin 12) dựa vào tên tài liệu đã cung cấp.
        3. Nếu thông tin có trong ngữ cảnh, hãy trích dẫn nguồn cuối câu trả lời theo định dạng [Tên_File.pdf - Trang X].
        4. Trả lời ngắn gọn, súc tích, giọng văn thân thiện với học sinh.
        
        [NGỮ CẢNH]:
        {context_text}
        """

        try:
            stream = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                stream=True,
                temperature=0.3,
                max_tokens=2000
            )
            unique_sources = sorted(list(set(sources)))
            return stream, unique_sources
        except Exception as e:
            return f"Lỗi kết nối AI: {str(e)}", []

# ==============================================================================
# 4. MAIN APPLICATION
# ==============================================================================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"⚠️ Lỗi thư viện: {IMPORT_ERROR}")
        st.info("Vui lòng chạy lệnh: pip install flashrank")
        st.stop()
        
    UIManager.inject_custom_css()
    UIManager.render_sidebar()
    UIManager.render_header()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! Mình là KTC Chatbot. Bạn cần hỗ trợ bài tập Tin học phần nào?"}]
    
    groq_client = RAGEngine.load_groq_client()
    
    if "vector_db" not in st.session_state or st.session_state.vector_db is None:
        with st.spinner("🚀 Đang khởi động hệ thống tri thức số..."):
            embeddings = RAGEngine.load_embedding_model()
            st.session_state.vector_db = RAGEngine.build_or_load_vector_db(embeddings)
            if st.session_state.vector_db:
                st.toast("✅ Đã tải xong dữ liệu!", icon="📚")

    for msg in st.session_state.messages:
        bot_avatar = AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"
        avatar = "🧑‍🎓" if msg["role"] == "user" else bot_avatar
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if len(st.session_state.messages) < 2:
        st.markdown("##### 💡 Gợi ý ôn tập:")
        cols = st.columns(3)
        prompt_btn = None
        if cols[0].button("🐍 Python: Số nguyên tố"):
            prompt_btn = "Viết chương trình Python nhập vào một số nguyên n và kiểm tra xem n có phải là số nguyên tố hay không. Giải thích code."
        if cols[1].button("🗃️ CSDL: Khóa chính"):
            prompt_btn = "Giải thích khái niệm Khóa chính (Primary Key) trong CSDL quan hệ và cho ví dụ minh họa."
        if cols[2].button("⚖️ Luật An ninh mạng"):
            prompt_btn = "Nêu các hành vi bị nghiêm cấm theo Luật An ninh mạng Việt Nam. Trích dẫn điều khoản nếu có."
        if prompt_btn:
            st.session_state.temp_input = prompt_btn
            st.rerun()

    if "temp_input" in st.session_state and st.session_state.temp_input:
        user_input = st.session_state.temp_input
        del st.session_state.temp_input
    else:
        user_input = st.chat_input("Nhập câu hỏi của bạn tại đây...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"):
            response_placeholder = st.empty()
            
            if not groq_client:
                st.error("❌ Chưa cấu hình API Key.")
            else:
                stream, sources = RAGEngine.generate_response(groq_client, st.session_state.vector_db, user_input)
                
                full_response = ""
                if isinstance(stream, str):
                    response_placeholder.error(stream)
                else:
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                
                if sources:
                    with st.expander("📚 Tài liệu tham khảo (Đã kiểm chứng)"):
                        for src in sources:
                            st.markdown(f"- 📖 *{src}*")
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()