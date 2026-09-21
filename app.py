try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    # Falls back to system sqlite3. If chromadb still fails to import with
    # a "sqlite3 version too old" error, make sure pysqlite3-binary is in
    # requirements.txt and that this shim ran before chromadb is imported
    # anywhere (it must be the very first thing in this file).
    pass

import os
import uuid
import tempfile
import shutil
import atexit
from pathlib import Path

import logging
logging.getLogger("pypdf").setLevel(logging.ERROR)

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from groq import Groq

try:
    import chromadb
except ImportError as e:
    raise ImportError(
        "chromadb is required for this app. Install it with `pip install chromadb`."
    ) from e

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BridgeAI — Data and Decisions",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. COMPLETE THEME OVERRIDES (Fixes all dark boxes)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global App Reset */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #F4F6F0 !important;
        color: #17281D !important;
    }

    /* Background Landscape Glow */
    .stApp {
        background:
            radial-gradient(ellipse at 50% 20%, rgba(255, 255, 255, 0.92) 0%, rgba(244, 246, 240, 0.95) 65%, rgba(226, 235, 222, 0.98) 100%) !important;
    }

    /* Top Capsule Navigation */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 24px;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 9999px;
        box-shadow: 0 4px 20px rgba(34, 66, 41, 0.08);
        margin: 0.5rem auto 2rem auto;
        max-width: 1100px;
    }

    .nav-logo {
        font-weight: 800;
        font-size: 1.35rem;
        color: #1C3825;
        letter-spacing: -0.5px;
    }

    .nav-menu {
        display: flex;
        gap: 1.8rem;
        align-items: center;
        font-size: 0.92rem;
        font-weight: 500;
        color: #556B5C;
    }

    .nav-pill-btn {
        background: #1C3825 !important;
        color: #FFFFFF !important;
        padding: 7px 20px !important;
        border-radius: 9999px !important;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 1.5rem 1rem;
        max-width: 950px;
        margin: 0 auto 1.5rem auto;
    }

    .hero-title {
        font-size: 3.6rem;
        font-weight: 700;
        line-height: 1.12;
        letter-spacing: -1.8px;
        color: #1C3825;
        margin-bottom: 1rem;
    }

    .hero-title span {
        color: #729E7D;
        font-weight: 400;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #435E4B;
        max-width: 650px;
        margin: 0 auto 1.5rem auto;
        line-height: 1.6;
    }

    /* ---------------- SIDEBAR LIGHT THEME FIXES ---------------- */
    section[data-testid="stSidebar"] {
        background-color: #E8EDE1 !important;
        border-right: 1px solid rgba(46, 80, 56, 0.12) !important;
    }

    section[data-testid="stSidebar"] * {
        color: #1A3323 !important;
    }

    /* Fix API Key input and model select box */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] div[data-baseweb="input"],
    section[data-testid="stSidebar"] div[data-baseweb="select"],
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #1A3323 !important;
        border-color: #BAC9B6 !important;
        border-radius: 12px !important;
    }

    /* ---------------- FIX FILE UPLOADER (NO MORE DARK BOX) ---------------- */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 2px dashed #9CB69B !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
    }

    [data-testid="stFileUploader"] section * {
        color: #1A3323 !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #EBF1E8 !important;
        color: #1A3323 !important;
        border: 1px solid #9CB69B !important;
        border-radius: 9999px !important;
        font-weight: 600 !important;
    }

    /* ---------------- FIX CHAT INPUT BOX (NO MORE DARK BAR) ---------------- */
    div[data-testid="stChatInput"] {
        background: transparent !important;
        border: none !important;
    }

    div[data-testid="stChatInput"] > div {
        background-color: #FFFFFF !important;
        border: 1.5px solid #BAC9B6 !important;
        border-radius: 9999px !important;
        box-shadow: 0 4px 20px rgba(34, 66, 41, 0.08) !important;
        padding: 4px 14px !important;
    }

    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #1A3323 !important;
        font-size: 0.95rem !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #6C8272 !important;
    }

    div[data-testid="stChatInput"] button {
        background-color: #1C3825 !important;
        color: #FFFFFF !important;
        border-radius: 50% !important;
        border: none !important;
    }

    /* Buttons */
    div[data-testid="stButton"] > button {
        background: #1C3825 !important;
        color: white !important;
        border-radius: 9999px !important;
        border: none !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(28, 56, 37, 0.2) !important;
    }

    div[data-testid="stButton"] > button:hover {
        background: #2E5839 !important;
    }

    /* Chat Messages */
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 18px !important;
        border: 1px solid rgba(255, 255, 255, 0.95) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
        padding: 1rem 1.4rem !important;
        margin-bottom: 0.8rem;
        color: #1C3825 !important;
    }

    /* Source Chips */
    .source-chip {
        display: inline-block;
        background: #E3EDE2;
        color: #1C3825;
        border: 1px solid #C4D9C2;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. TOP NAVIGATION & HERO SECTION
# -----------------------------------------------------------------------------
st.markdown("""
<div class="top-nav">
    <div class="nav-logo">BridgeAI<span style="font-size:0.75rem; vertical-align:super;">™</span></div>
    <div class="nav-menu">
        <span style="color: #1C3825; font-weight:700; text-decoration: underline; text-underline-offset: 4px;">Mission</span>
        <span>How it Works</span>
        <span>Pricing</span>
        <a href="#chat" class="nav-pill-btn">Book a Demo</a>
    </div>
    <div style="font-size: 0.85rem; color: #556B5C; font-weight:500;">
        New Account &nbsp;|&nbsp; <strong>Login</strong>
    </div>
</div>

<div class="hero-container">
    <div class="hero-title">
        Bridge the gap <span>between</span><br>data and decisions
    </div>
    <div class="hero-subtitle">
        Turn disconnected data and documents into actionable insights with AI-powered automation.
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. EMBEDDINGS MODEL
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

embed_model = load_embedding_model()

# -----------------------------------------------------------------------------
# 5. SESSION-SCOPED STORAGE
#
# Each browser session gets its own Chroma persist directory and its own
# collection name. This prevents one user's "Index Documents" click from
# deleting or overwriting another user's indexed data when the app is
# deployed for more than one concurrent user, and avoids stale collections
# piling up on disk across app restarts.
# -----------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

if "chroma_dir" not in st.session_state:
    chroma_dir = os.path.join(tempfile.gettempdir(), f"bridgeai_chroma_{st.session_state.session_id}")
    os.makedirs(chroma_dir, exist_ok=True)
    st.session_state.chroma_dir = chroma_dir

    # Best-effort cleanup when the Python process exits. This won't fire on
    # every Streamlit deployment model (e.g. some hosts kill workers hard),
    # but it prevents obvious leakage during normal local/dev usage.
    def _cleanup(path=chroma_dir):
        shutil.rmtree(path, ignore_errors=True)
    atexit.register(_cleanup)

# -----------------------------------------------------------------------------
# 6. VECTOR STORE & RETRIEVER
# -----------------------------------------------------------------------------
class VectorStore:
    def __init__(self, persist_directory: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Fresh collection name is generated per index run (see below), so
        # there is nothing to delete here — no risk of clobbering another
        # session's or another run's data.
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents, embeddings, batch_size=200):
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start:start + batch_size]
            batch_embs = embeddings[start:start + batch_size]
            ids, texts, metas, embs = [], [], [], []
            for offset, (doc, emb) in enumerate(zip(batch_docs, batch_embs)):
                ids.append(f"chunk_{start + offset}_{uuid.uuid4().hex[:8]}")
                texts.append(doc.page_content)
                metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    elif value is not None:
                        metadata[key] = str(value)
                metas.append(metadata or {"source_file": "document.pdf"})
                embs.append(list(emb))
            self.collection.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)


class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embed_model):
        self.vector_store = vector_store
        self.embed_model = embed_model

    def retrieve(self, query: str, top_k: int = 4, score_threshold: float = 0.0):
        query = query.strip()
        if not query or self.vector_store.collection.count() == 0:
            return []

        embeddings = list(self.embed_model.embed([query]))
        if not embeddings:
            return []
        query_embedding = embeddings[0].tolist() if hasattr(embeddings[0], "tolist") else list(embeddings[0])

        result = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.vector_store.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        found = []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        for rank, (doc_id, text, metadata, distance) in enumerate(zip(ids, docs, metas, distances), 1):
            score = 1.0 - float(distance)
            if score >= score_threshold:
                found.append({
                    "id": doc_id,
                    "content": text,
                    "metadata": metadata or {},
                    "similarity_score": score,
                    "rank": rank
                })
        return found

# -----------------------------------------------------------------------------
# 7. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
# NOTE ON MODELS: mixtral-8x7b-32768 has long been removed from Groq, and
# both llama-3.1-8b-instant and llama-3.3-70b-versatile were deprecated and
# shut down by Groq on 2026-08-16. Using any of the three original model IDs
# will make every chat request fail. Current, non-deprecated options below
# per https://console.groq.com/docs/deprecations and /docs/models — re-check
# that page periodically, since Groq's lineup changes often.
GROQ_MODEL_OPTIONS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]

with st.sidebar:
    st.markdown("### 🌿 **BridgeCore™ Engine**")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get your key at https://console.groq.com"
    )

    model_name = st.selectbox(
        "LLM Model",
        options=GROQ_MODEL_OPTIONS,
        index=0,
        help="List trimmed to models Groq currently serves. Check "
             "console.groq.com/docs/models if you need something else."
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    st.markdown("---")
    st.markdown("#### 📁 **Document Repository**")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    with st.expander("⚙️ Advanced Parameters"):
        top_k = st.slider("Passages (Top-K)", min_value=1, max_value=8, value=4)
        min_score = st.slider("Min Relevance", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        chunk_size = st.number_input("Chunk Size", min_value=100, value=900, step=100)
        chunk_overlap = st.number_input("Overlap", min_value=0, value=150, step=25)

    process_btn = st.button("Index Documents", use_container_width=True)

# -----------------------------------------------------------------------------
# 8. DOCUMENT INGESTION
# -----------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.retriever = None

if process_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF file.")
    elif chunk_overlap >= chunk_size:
        st.sidebar.error("Overlap must be smaller than Chunk Size.")
    else:
        with st.spinner("Processing & Indexing PDFs..."):
            all_pages = []
            with tempfile.TemporaryDirectory() as tmp_dir:
                for uploaded_file in uploaded_files:
                    temp_path = Path(tmp_dir) / uploaded_file.name
                    temp_path.write_bytes(uploaded_file.getvalue())

                    try:
                        loader = PyPDFLoader(str(temp_path))
                        pages = loader.load()
                        for p in pages:
                            p.metadata["source_file"] = uploaded_file.name
                        all_pages.extend(pages)
                    except Exception as e:
                        st.error(f"Error reading {uploaded_file.name}: {e}")

            # Drop pages with no extractable text (e.g. scanned/image-only
            # PDFs with no OCR layer) before they reach the splitter/embedder.
            usable_pages = [p for p in all_pages if p.page_content and p.page_content.strip()]
            skipped = len(all_pages) - len(usable_pages)

            if not all_pages:
                st.sidebar.error("No pages could be read from the uploaded file(s).")
            elif not usable_pages:
                st.sidebar.error(
                    "No extractable text found in the uploaded PDF(s). "
                    "They may be scanned images without an OCR text layer."
                )
            else:
                if skipped:
                    st.sidebar.warning(f"Skipped {skipped} page(s) with no extractable text.")

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = splitter.split_documents(usable_pages)
                chunks = [c for c in chunks if c.page_content and c.page_content.strip()]

                if not chunks:
                    st.sidebar.error("Splitting produced no usable chunks.")
                else:
                    texts = [doc.page_content for doc in chunks]
                    embeddings = list(embed_model.embed(texts))

                    # Fresh collection name per index run avoids any collision
                    # with a previous run's data still being read by a
                    # slow-finishing request elsewhere in the same session.
                    collection_name = f"bridge_docs_{uuid.uuid4().hex[:8]}"
                    vectorstore = VectorStore(st.session_state.chroma_dir, collection_name)
                    vectorstore.add_documents(chunks, embeddings)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.retriever = RAGRetriever(vectorstore, embed_model)
                    st.sidebar.success(f"🌿 Indexed {len(chunks)} chunks from {len(usable_pages)} pages!")

# -----------------------------------------------------------------------------
# 9. CHAT INTERFACE
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            chips_html = "".join(
                f'<span class="source-chip">📄 {s["source"]} (p. {s["page"]}) — {int(s["similarity"] * 100)}%</span>'
                for s in message["sources"]
            )
            st.markdown(f"<div style='margin-top:6px;'>{chips_html}</div>", unsafe_allow_html=True)

# Chat Input Box
if prompt := st.chat_input("Ask any question grounded in your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.retriever:
            answer = "Please upload and index PDF documents first in the sidebar."
            sources = []
            st.warning(answer)
        else:
            retriever = st.session_state.retriever
            results = retriever.retrieve(prompt, top_k=top_k, score_threshold=min_score)

            if not results:
                answer = "I couldn't find any relevant answers in your indexed documents."
                sources = []
                st.markdown(answer)
            else:
                context = "\n\n".join(
                    f"[Source: {r['metadata'].get('source_file', 'unknown')}, Page: {r['metadata'].get('page', 'unknown')}]\n{r['content']}"
                    for r in results
                )

                sources = [
                    {
                        "source": r["metadata"].get("source_file", "document.pdf"),
                        "page": (r["metadata"].get("page", 0) + 1) if isinstance(r["metadata"].get("page"), int) else r["metadata"].get("page", "1"),
                        "similarity": round(r["similarity_score"], 3)
                    }
                    for r in results
                ]

                if not groq_api_key.strip():
                    answer = "⚠️ **Groq API Key is missing.** Please provide your key in the sidebar."
                    sources = []
                    st.markdown(answer)
                else:
                    try:
                        client = Groq(api_key=groq_api_key)

                        system_prompt = (
                            "You are BridgeAI, an enterprise document intelligence assistant. "
                            "Answer questions truthfully and accurately using ONLY the provided context. "
                            "If the context does not contain the answer, say that you don't have enough information."
                        )
                        user_content = f"Context:\n{context}\n\nQuestion:\n{prompt}"

                        completion = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_content}
                            ],
                            temperature=temperature,
                            max_tokens=1024
                        )

                        choices = getattr(completion, "choices", None)
                        if not choices:
                            answer = "⚠️ Groq returned an empty response. Please try again."
                            sources = []
                        else:
                            answer = choices[0].message.content or "⚠️ Groq returned an empty message."

                        st.markdown(answer)

                    except Exception as e:
                        answer = f"❌ Groq Error: {e}"
                        sources = []
                        st.error(answer)

                if sources:
                    chips_html = "".join(
                        f'<span class="source-chip">📄 {s["source"]} (p. {s["page"]}) — {int(s["similarity"] * 100)}%</span>'
                        for s in sources
                    )
                    st.markdown(f"<div style='margin-top:6px;'>{chips_html}</div>", unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
