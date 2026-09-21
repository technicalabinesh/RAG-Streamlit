import os
import uuid
import tempfile
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BridgeAI — Data & Document Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. BRIDGEAI STYLING & DESIGN THEME (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,500;0,600;1,400&display=swap');

    :root {
        --primary-green: #224229;
        --accent-green: #3B6B48;
        --light-sage: #8DAA91;
        --bg-warm: #F4F6F0;
        --card-bg: rgba(255, 255, 255, 0.75);
        --text-dark: #19241C;
        --text-muted: #526356;
        --pill-bg: #FFFFFF;
    }

    /* Base App Styling */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #E7EDE0 0%, #F5F7F2 60%, #EBF0E6 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-dark);
    }

    /* Top Capsule Navigation */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 24px;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.6);
        border-radius: 9999px;
        box-shadow: 0 4px 20px rgba(34, 66, 41, 0.06);
        margin-bottom: 2rem;
    }

    .nav-logo {
        font-weight: 800;
        font-size: 1.3rem;
        color: var(--primary-green);
        letter-spacing: -0.5px;
    }

    .nav-menu {
        display: flex;
        gap: 1.5rem;
        align-items: center;
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--text-muted);
    }

    .nav-pill-btn {
        background: var(--primary-green) !important;
        color: #fff !important;
        padding: 6px 18px !important;
        border-radius: 9999px !important;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }

    .nav-pill-btn:hover {
        background: var(--accent-green) !important;
        transform: translateY(-1px);
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 2rem 1rem;
        max-width: 900px;
        margin: 0 auto;
    }

    .hero-title {
        font-size: 3.4rem;
        font-weight: 700;
        line-height: 1.15;
        letter-spacing: -1.5px;
        color: var(--primary-green);
        margin-bottom: 1rem;
    }

    .hero-title span {
        color: var(--light-sage);
        font-weight: 400;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: var(--text-muted);
        max-width: 620px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
        font-weight: 400;
    }

    /* Glass Cards */
    .glass-card {
        background: var(--card-bg);
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 10px 30px rgba(34, 66, 41, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* Streamlit Custom UI Overrides */
    .stSidebar {
        background-color: #EBF0E6 !important;
        border-right: 1px solid rgba(34, 66, 41, 0.08);
    }

    div[data-testid="stButton"] > button {
        background-color: var(--primary-green) !important;
        color: white !important;
        border-radius: 9999px !important;
        border: none !important;
        padding: 0.55rem 1.4rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.2px;
        transition: all 0.2s ease-in-out;
    }

    div[data-testid="stButton"] > button:hover {
        background-color: var(--accent-green) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(34, 66, 41, 0.2);
    }

    /* Chat Messages Styling */
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.8) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.9) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        margin-bottom: 0.8rem;
    }

    .source-chip {
        display: inline-block;
        background: #E2ECE1;
        color: var(--primary-green);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. TOP NAVIGATION & HERO BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="top-nav">
    <div class="nav-logo">BridgeAI<span style="font-size:0.8rem; vertical-align:super;">™</span></div>
    <div class="nav-menu">
        <span style="color: #224229; font-weight:600; text-decoration:underline;">Mission</span>
        <span>How it Works</span>
        <span>Pricing</span>
        <a href="#chat" class="nav-pill-btn">Ask Question</a>
    </div>
    <div style="font-size: 0.85rem; color: #526356; font-weight:500;">
        PDF Workspace &nbsp;→&nbsp; <span style="font-weight:600; color:#224229;">Active</span>
    </div>
</div>

<div class="hero-container">
    <div class="hero-title">
        Bridge the gap <span>between</span><br>data and decisions
    </div>
    <div class="hero-subtitle">
        Turn disconnected PDF data and documents into actionable insights with AI-powered document intelligence.
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. CACHED EMBEDDINGS MODEL
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# -----------------------------------------------------------------------------
# 5. VECTOR STORE & RETRIEVER CLASSES
# -----------------------------------------------------------------------------
class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "pdf_docs"):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_directory)
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents, embeddings, batch_size=500):
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start:start + batch_size]
            batch_embs = embeddings[start:start + batch_size]
            ids, texts, metas, embs = [], [], [], []
            for offset, (doc, emb) in enumerate(zip(batch_docs, batch_embs)):
                ids.append(f"chunk_{start+offset}_{uuid.uuid4().hex[:8]}")
                texts.append(doc.page_content)
                metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    elif value is not None:
                        metadata[key] = str(value)
                metas.append(metadata or {"source_file": "unknown"})
                embs.append(emb.tolist())
            self.collection.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, model: SentenceTransformer):
        self.vector_store = vector_store
        self.model = model

    def retrieve(self, query: str, top_k: int = 4, score_threshold: float = 0.0):
        if not query.strip() or self.vector_store.collection.count() == 0:
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)[0].tolist()
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
# 6. SIDEBAR CONFIGURATION (CHATGPT / OPENAI)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌿 **BridgeCore™ Setup**")
    
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Enter your OpenAI API key starting with 'sk-'"
    )

    chatgpt_model = st.selectbox(
        "ChatGPT Engine",
        options=[
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ],
        index=0
    )

    temperature = st.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    st.markdown("---")
    st.markdown("#### 📁 **Document Repository**")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    with st.expander("⚙️ Advanced Retrieval Parameters"):
        top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=8, value=4)
        min_score = st.slider("Min Relevance Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        chunk_size = st.number_input("Chunk Size", value=900, step=100)
        chunk_overlap = st.number_input("Chunk Overlap", value=150, step=25)

    process_btn = st.button("Index Documents", use_container_width=True)

# -----------------------------------------------------------------------------
# 7. DOCUMENT INGESTION PIPELINE
# -----------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.retriever = None

if process_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF.")
    else:
        with st.spinner("Analyzing & Indexing documents..."):
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
                        st.error(f"Error loading {uploaded_file.name}: {e}")

            if all_pages:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = splitter.split_documents(all_pages)
                texts = [doc.page_content for doc in chunks]
                embeddings = embedding_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

                vectorstore = VectorStore()
                vectorstore.add_documents(chunks, embeddings)

                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = RAGRetriever(vectorstore, embedding_model)
                st.sidebar.success(f"🌿 Successfully indexed {len(chunks)} chunks from {len(all_pages)} pages!")

# -----------------------------------------------------------------------------
# 8. CHAT CONVERSATION INTERFACE
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            chips_html = "".join(
                [f'<span class="source-chip">📄 {s["source"]} (p. {s["page"]}) — {int(s["similarity"]*100)}%</span>' for s in message["sources"]]
            )
            st.markdown(f"<div style='margin-top:8px;'>{chips_html}</div>", unsafe_allow_html=True)

# User Query input
if prompt := st.chat_input("Ask any question grounded in your PDF documents..."):
    if not st.session_state.retriever:
        st.warning("Please upload and index documents in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            retriever = st.session_state.retriever
            results = retriever.retrieve(prompt, top_k=top_k, score_threshold=min_score)

            if not results:
                answer = "I couldn't find any relevant sections in your indexed files to answer that."
                sources = []
                st.markdown(answer)
            else:
                context = "\n\n".join(
                    f"[Source: {r['metadata'].get('source_file', 'unknown')}, Page: {r['metadata'].get('page', 'unknown')}]\n{r['content']}"
                    for r in results
                )

                sources = [
                    {
                        "source": r["metadata"].get("source_file", "unknown"),
                        "page": r["metadata"].get("page", 0) + 1 if isinstance(r["metadata"].get("page"), int) else r["metadata"].get("page", "1"),
                        "similarity": round(r["similarity_score"], 3)
                    }
                    for r in results
                ]

                if not openai_api_key.strip():
                    answer = (
                        "⚠️ **OpenAI API Key is missing.** Passages retrieved, but ChatGPT answer generation requires a key in the sidebar."
                    )
                    st.markdown(answer)
                else:
                    try:
                        llm = ChatOpenAI(
                            api_key=openai_api_key,
                            model=chatgpt_model,
                            temperature=temperature,
                            max_tokens=1000
                        )

                        rag_prompt = f"""You are BridgeAI, a professional enterprise knowledge assistant. Answer the user question accurately using ONLY the context provided below. If the answer is not present, state that clearly without guessing.

Context:
{context}

Question: {prompt}

Answer:"""
                        response = llm.invoke(rag_prompt)
                        answer = response.content
                        st.markdown(answer)

                    except Exception as e:
                        answer = f"❌ OpenAI Error: {e}"
                        st.error(answer)

                if sources:
                    chips_html = "".join(
                        [f'<span class="source-chip">📄 {s["source"]} (p. {s["page"]}) — {int(s["similarity"]*100)}%</span>' for s in sources]
                    )
                    st.markdown(f"<div style='margin-top:8px;'>{chips_html}</div>", unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
