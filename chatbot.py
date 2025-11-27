# streamlit_app.py
import os
import glob
import time
from typing import List, Optional

import streamlit as st
from pypdf import PdfReader

# AI / RAG libs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Translator (HuggingFace)
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Groq client for LLM streaming (as in your original)
from groq import Groq

# --- 0. CẤU HÌNH CHUNG ---
st.set_page_config(
    page_title="KTC Assistant - Trợ lý Tin học 2025",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CONSTANTS = {
    "MODEL_NAME": 'llama-3.1-8b-instant',
    "PDF_DIR": "./PDF_KNOWLEDGE",
    "VECTOR_STORE_PATH": "./faiss_db_index",
    "LOGO_PATH": "LOGO.jpg",
    # Mô hình embedding hỗ trợ nhiều ngôn ngữ, tốt cho vi/en cross-lingual
    "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # Dịch Vi -> En (có thể đổi sang model khác nếu muốn)
    "TRANSLATION_MODEL": "Helsinki-NLP/opus-mt-vi-en",
    "CHUNK_SIZE": 800,
    "CHUNK_OVERLAP": 150,
    "TOP_K": 3,
}

# --- 1. CSS / Giao diện cơ bản ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
    .stApp {background-color: #f8f9fa;}
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    .gradient-text {
        background: linear-gradient(90deg, #0052cc, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        text-align: center;
        padding: 10px 0;
    }
    .source-box {
        font-size: 0.85rem; color: #444; background: #f1f1f1;
        padding: 8px; border-radius: 6px; margin-top: 8px; border-left: 3px solid #0284c7;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. CACHE / TÀI NGUYÊN DÙNG CHUNG ---
@st.cache_resource(show_spinner=False)
def get_groq_client():
    """Load Groq client 1 lần."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=api_key)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Tạo HuggingFaceEmbeddings 1 lần và cache lại."""
    return HuggingFaceEmbeddings(model_name=CONSTANTS["EMBEDDING_MODEL"])

@st.cache_resource(show_spinner=False)
def get_translator():
    """Tải model dịch (vi->en) một lần. Trả về pipeline dịch."""
    model_name = CONSTANTS["TRANSLATION_MODEL"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("translation", model=model, tokenizer=tokenizer, src_lang="vi", tgt_lang="en")

# --- 3. CLASS KnowledgeBase (đã cải tiến) ---
class KnowledgeBase:
    """Quản lý đọc file, tách văn bản, tạo/ load vector DB."""
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def load_documents(self) -> List[Document]:
        """Đọc tất cả PDF trong thư mục và trả về danh sách Document (langchain)."""
        if not os.path.exists(CONSTANTS["PDF_DIR"]):
            os.makedirs(CONSTANTS["PDF_DIR"])
            return []

        pdf_files = glob.glob(os.path.join(CONSTANTS["PDF_DIR"], "*.pdf"))
        documents: List[Document] = []
        for pdf_path in pdf_files:
            try:
                reader = PdfReader(pdf_path)
                file_name = os.path.basename(pdf_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": file_name, "page": i + 1}
                        ))
            except Exception as e:
                st.warning(f"Không đọc được file {pdf_path}: {e}")
        return documents

    def build_or_load_vector_db(self, force_rebuild: bool = False) -> Optional[FAISS]:
        """
        Nếu đã có lưu trên disk -> load, nếu không -> build mới.
        Dùng cache_resource ở mức gọi hàm này để tránh build lại nhiều lần.
        """
        path = CONSTANTS["VECTOR_STORE_PATH"]
        # Try load
        if os.path.exists(path) and not force_rebuild:
            try:
                return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            except Exception:
                # nếu load lỗi --> build lại
                pass

        # Build mới
        docs = self.load_documents()
        if not docs:
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONSTANTS["CHUNK_SIZE"],
            chunk_overlap=CONSTANTS["CHUNK_OVERLAP"]
        )
        splits = text_splitter.split_documents(docs)
        if not splits:
            return None

        vector_db = FAISS.from_documents(splits, self.embeddings)
        # Lưu xuống disk
        try:
            os.makedirs(path, exist_ok=True)
            vector_db.save_local(path)
        except Exception as e:
            st.warning(f"Không lưu được vector DB: {e}")
        return vector_db

# --- 4. Các hàm tiện ích (translate + search + format) ---
def translate_vi_to_en(translator_pipeline, text: str) -> str:
    """Dịch tiếng Việt sang tiếng Anh. Trả về chuỗi tiếng Anh."""
    if not text or not translator_pipeline:
        return text
    try:
        result = translator_pipeline(text, max_length=512)
        # pipeline trả dict hoặc list dict
        if isinstance(result, list):
            return result[0]["translation_text"]
        return result.get("translation_text", text)
    except Exception:
        # nếu dịch lỗi -> fallback trả về text gốc
        return text

def retrieve_context(vector_db: FAISS, query_en: str, k: int = CONSTANTS["TOP_K"]):
    """Tìm kiếm tương tự (similarity search) trên vector DB bằng query tiếng Anh.
       Trả về context (chuỗi) và list nguồn."""
    if not vector_db or not query_en:
        return "", []
    try:
        docs = vector_db.similarity_search(query_en, k=k)
        context_parts = []
        sources = []
        for d in docs:
            txt = d.page_content.strip()
            meta = d.metadata or {}
            src = f"{meta.get('source', 'unknown')} (Tr. {meta.get('page', '?')})"
            context_parts.append(f"[TRÍCH]: {txt}")
            sources.append(src)
        context_text = "\n\n".join(context_parts)
        return context_text, sources
    except Exception as e:
        st.warning(f"Lỗi khi truy vấn vector DB: {e}")
        return "", []

def build_system_prompt(context_en: str) -> str:
    """Tạo system prompt cho LLM: dữ liệu ngữ cảnh ở dạng tiếng Anh,
       yêu cầu LLM trả lời bằng tiếng Việt."""
    system = f"""
Bạn là trợ lý ảo KTC, chuyên gia môn Tin học theo chương trình GDPT 2018.
NHIỆM VỤ: Trả lời câu hỏi người dùng dựa trên phần [THÔNG TIN TÀI LIỆU] bên dưới.
QUY TẮC:
1) Chỉ sử dụng thông tin có trong [THÔNG TIN TÀI LIỆU]. Nếu không tìm thấy, nói rõ: "SGK hiện chưa đề cập vấn đề này."
2) Viết bằng tiếng Việt chuẩn, thân thiện, sư phạm, phù hợp học sinh.
3) Nếu trích dẫn tài liệu, liệt kê nguồn (tên file và số trang).
4) Trả lời ngắn gọn, có cấu trúc: tiêu đề in đậm, các bước / gạch đầu dòng khi cần.
[THÔNG TIN TÀI LIỆU - ENGLISH]:
{context_en}
    """
    return system

# --- 5. Khởi tạo tài nguyên (cached) ---
groq_client = get_groq_client()
if groq_client is None:
    st.error("⚠️ Lỗi: Chưa cấu hình GROQ_API_KEY trong secrets.")
    st.stop()

embeddings = get_embeddings()
translator = get_translator()
kb = KnowledgeBase(embeddings)

# Load / Build vector DB - NOTE: dùng cached hàm ở trên để tránh build lại liên tục
if "vector_db" not in st.session_state:
    with st.spinner("🔄 Khởi tạo hệ tri thức..."):
        st.session_state.vector_db = kb.build_or_load_vector_db(force_rebuild=False)

# --- 6. Sidebar (Control) ---
with st.sidebar:
    if os.path.exists(CONSTANTS["LOGO_PATH"]):
        st.image(CONSTANTS["LOGO_PATH"], use_container_width=True)
    st.title("⚙️ Control Panel")
    status_color = "green" if st.session_state.vector_db else "red"
    status_text = "Đã nạp kiến thức" if st.session_state.vector_db else "Chưa có dữ liệu"
    st.markdown(f"**Trạng thái:** <span style='color:{status_color}'>● {status_text}</span>", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🔄 Rebuild Vector DB (đọc lại PDF)"):
        with st.spinner("Đang build lại vector DB — có thể mất vài phút..."):
            st.session_state.vector_db = kb.build_or_load_vector_db(force_rebuild=True)
        st.success("✅ Đã rebuild xong.")
        time.sleep(0.5)
        st.experimental_rerun()

    if st.button("🗑️ Xóa lịch sử Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

    st.markdown("""
    <div style="background:#f8f9fa; padding:12px; border-radius:8px; border:1px dashed #ccc; margin-top:10px;">
        <div style="font-weight:bold; color:#0052cc;">🚀 DỰ ÁN KHKT 2025-2026</div>
        <div style="font-size:0.9rem;">GVHD: <b>Thầy Nguyễn Thế Khanh</b></div>
        <div style="font-size:0.9rem;">Học sinh: <b>Bùi Tá Tùng - Cao Sỹ Bảo Chung</b></div>
    </div>
    """, unsafe_allow_html=True)

# --- 7. Session state cho messages (chat history) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là **KTC AI**. Hãy hỏi tôi bất cứ điều gì về Tin học trong SGK."}
    ]

# --- 8. MAIN UI: hiển thị chat và xử lý input ---
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    st.markdown('<h1 class="gradient-text">TRỢ LÝ ẢO TIN HỌC KTC</h1>', unsafe_allow_html=True)

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        avatar = "🧑‍🎓" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Input
    prompt = st.chat_input("Bạn muốn tìm hiểu gì về Tin học? (gõ tiếng Việt)")

    if prompt:
        # 1) Append user message & show ngay
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        # 2) Preprocess: dịch sang tiếng Anh để truy vấn vector DB
        with st.spinner("🔎 Đang tìm kiếm thông tin liên quan..."):
            # Dịch câu hỏi Vi -> En để search
            query_en = translate_vi_to_en(translator, prompt)
            context_en, sources = retrieve_context(st.session_state.vector_db, query_en, k=CONSTANTS["TOP_K"])

        # 3) Build system prompt (context bằng tiếng Anh) và gọi LLM streaming
        system_prompt = build_system_prompt(context_en)

        # Placeholder cho streaming trả về
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Chuẩn bị messages: system + recent history (khoảng 4 turn cuối)
                messages_for_model = [
                    {"role": "system", "content": system_prompt},
                    # gửi câu hỏi gốc tiếng Việt dưới dạng user message (để model biết yêu cầu trả lời tiếng Việt)
                    {"role": "user", "content": f"Original question (VN): {prompt}"},
                    {"role": "user", "content": f"Search query used (EN): {query_en}"}
                ]
                # Dùng Groq streaming
                stream = groq_client.chat.completions.create(
                    messages=messages_for_model,
                    model=CONSTANTS["MODEL_NAME"],
                    stream=True,
                    temperature=0.2,
                    max_tokens=1024
                )

                # Hiển thị streaming: cập nhật message_placeholder dần dần
                for chunk in stream:
                    # chunk có cấu trúc tương tự OpenAI streaming deltas
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        piece = delta.content
                        full_response += piece
                        # Hiển thị kèm con trỏ
                        message_placeholder.markdown(full_response + "▌")
                # Sau streaming kết thúc: thêm nguồn nếu có
                if sources:
                    unique_sources = list(dict.fromkeys(sources))  # giữ thứ tự, loại trùng
                    sources_html = "<div class='source-box'>📚 <b>Nguồn tham khảo:</b><br>" + "<br>".join([f"• {s}" for s in unique_sources]) + "</div>"
                    final = full_response + "\n\n" + sources_html
                    message_placeholder.markdown(final, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": final})
                else:
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                # Nếu streaming không được, hiển thị lỗi thân thiện
                err_msg = f"Đã xảy ra lỗi khi gọi AI: {e}"
                message_placeholder.markdown(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
