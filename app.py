import os
import uuid
import tempfile
import streamlit as st
import numpy as np
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Assistant")
st.caption("Chat with your PDF documents using SentenceTransformers, ChromaDB, and Groq LLMs.")

# -----------------------------------------------------------------------------
# CACHED RESOURCES
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Embedding Model...")
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

embedding_model = get_embedding_model()

# -----------------------------------------------------------------------------
# CORE RAG CLASSES
# -----------------------------------------------------------------------------
class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "pdf_docs"):
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

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
        if not query.strip():
            return []
        count = self.vector_store.collection.count()
        if count == 0:
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)[0].tolist()
        result = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"]
        )

        found = []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        for rank, (doc_id, text, metadata, distance) in enumerate(zip(ids, docs, metas, distances), 1):
            score = 1.0 - float(distance)  # Cosine distance to similarity
            if score >= score_threshold:
                found.append({
                    "id": doc_id,
                    "content": text,
                    "metadata": metadata or {},
                    "similarity_score": score,
                    "distance": float(distance),
                    "rank": rank
                })
        return found

# -----------------------------------------------------------------------------
# SIDEBAR SETTINGS & FILE UPLOAD
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get your key at https://console.groq.com"
    )

    model_name = st.selectbox(
        "Groq Model",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ],
        index=0
    )

    st.divider()
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.subheader("🔍 Retrieval Settings")
    top_k = st.slider("Top-K Passages", min_value=1, max_value=10, value=4)
    min_score = st.slider("Min Similarity Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    chunk_size = st.number_input("Chunk Size", value=1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap", value=200, step=50)

    process_btn = st.button("Index Documents", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# DOCUMENT PROCESSING LOGIC
# -----------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.retriever = None

if process_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF file.")
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
                st.sidebar.success(f"Indexed {len(chunks)} chunks from {len(all_pages)} pages!")

# -----------------------------------------------------------------------------
# CHAT INTERFACE
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources & Similarity"):
                for s in message["sources"]:
                    st.write(f"- **{s['source']}** (Page {s['page']}) — Score: `{s['similarity']}`")

# User prompt input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.retriever:
        st.warning("Please upload and index documents before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            retriever = st.session_state.retriever
            results = retriever.retrieve(prompt, top_k=top_k, score_threshold=min_score)

            if not results:
                answer = "I couldn't find any relevant information in the indexed documents."
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
                        "page": r["metadata"].get("page", 0) + 1 if isinstance(r["metadata"].get("page"), int) else r["metadata"].get("page", "unknown"),
                        "similarity": round(r["similarity_score"], 4)
                    }
                    for r in results
                ]

                if not groq_api_key.strip():
                    answer = (
                        "⚠️ **Groq API Key is missing.** Passages were retrieved successfully, "
                        "but LLM answer generation is disabled."
                    )
                    st.markdown(answer)
                else:
                    try:
                        llm = ChatGroq(
                            api_key=groq_api_key,
                            model=model_name,
                            temperature=0.1,
                            max_tokens=1024
                        )

                        rag_prompt = f"""You are a helpful assistant. Answer using only the supplied document context.
If the context does not contain the answer, clearly say that you cannot determine it from these documents.

Context:
{context}

Question: {prompt}

Answer:"""
                        response = llm.invoke(rag_prompt)
                        answer = response.content
                        st.markdown(answer)

                    except Exception as e:
                        answer = f"❌ Error communicating with Groq: {e}"
                        st.error(answer)

                if sources:
                    with st.expander("📚 View Sources & Similarity"):
                        for s in sources:
                            st.write(f"- **{s['source']}** (Page {s['page']}) — Score: `{s['similarity']}`")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })