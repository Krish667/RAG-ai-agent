import os
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOC_DIR = Path("doc_files")
DB_DIR = "chroma_db"
LOG_FILE = Path("app_data.json")
DOC_DIR.mkdir(exist_ok=True)

if not LOG_FILE.exists():
    LOG_FILE.write_text(json.dumps({"documents": [], "history": []}))


def load_data():
    return json.loads(LOG_FILE.read_text())


def save_data(data):
    LOG_FILE.write_text(json.dumps(data, indent=2))


def get_vector_store():
    return Chroma(
        collection_name="documents",
        embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
        persist_directory=DB_DIR,
    )


def process_document(file_path):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        docs = PyPDFLoader(str(file_path)).load()
    elif ext == ".docx":
        docs = Docx2txtLoader(str(file_path)).load()
    else:
        docs = TextLoader(str(file_path), encoding="utf-8").load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)

    for doc in docs:
        doc.metadata["source"] = file_path.name

    get_vector_store().add_documents(docs)


def get_text(content):
    """Gemini kabhi string, kabhi content-blocks ki list return karta hai; dono handle karta hai."""
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in content
        if isinstance(block, str) or block.get("type") == "text"
    )


def answer_question(query):
    db = get_vector_store()
    docs = db.similarity_search(query, k=4)

    context = "\n\n".join(
        f"SOURCE: {d.metadata.get('source', 'Unknown')} | "
        f"PAGE: {d.metadata.get('page', 0) + 1}\n{d.page_content}"
        for d in docs
    )

    prompt = f"""You are a document assistant. Answer only from the provided context.
If the answer is not present, say that it was not found in the uploaded documents.
Keep the answer clear and concise.

CONTEXT:
{context}

QUESTION: {query}
"""
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.7-flash"),
        temperature=0,
    )
    answer = get_text(llm.invoke(prompt).content)
    return answer, docs


st.set_page_config(page_title="Document Intelligence Assistant", layout="wide")
st.title("📄 AI Document Intelligence Assistant")
st.caption("RAG-powered assistant for PDF, DOCX and TXT documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: upload + admin
with st.sidebar:
    st.header("Document Management")
    files = st.file_uploader(
        "Upload documents", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )

    if files and st.button("Process Documents"):
        with st.spinner("Processing..."):
            data = load_data()
            existing = {d["name"] for d in data["documents"]}
            for file in files:
                path = DOC_DIR / file.name
                path.write_bytes(file.getvalue())
                if file.name not in existing:
                    process_document(path)
                    data["documents"].append({"name": file.name, "size": file.size})
            save_data(data)
        st.success("Documents processed.")
        st.rerun()

    st.divider()
    st.header("Admin Dashboard")
    data = load_data()
    st.metric("Documents", len(data["documents"]))
    st.metric("Queries", len(data["history"]))

    if data["documents"]:
        selected = st.selectbox("Delete document", [d["name"] for d in data["documents"]])
        if st.button("Delete Selected"):
            get_vector_store().delete(where={"source": selected})
            path = DOC_DIR / selected
            if path.exists():
                path.unlink()
            data["documents"] = [d for d in data["documents"] if d["name"] != selected]
            save_data(data)
            st.success("Document deleted.")
            st.rerun()

    if st.checkbox("Show response history"):
        for item in reversed(data["history"][-10:]):
            st.write(f"**Q:** {item['question']}")
            st.write(f"**A:** {item['answer']}")

# Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask anything related to uploaded documents...")
if query:
    if not load_data()["documents"]:
        st.warning("Please upload a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                answer, docs = answer_question(query)
            st.markdown(answer)
            st.markdown("**Sources:**")
            seen = set()
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page")
                label = f"{source}" + (f" — page {page + 1}" if page is not None else "")
                if label not in seen:
                    st.caption(label)
                    seen.add(label)

        source_text = ", ".join(
            f"{d.metadata.get('source', 'Unknown')}"
            + (f" (page {d.metadata.get('page') + 1})" if d.metadata.get('page') is not None else "")
            for d in docs
        )
        st.session_state.messages.append({"role": "assistant", "content": answer})
        data = load_data()
        data["history"].append({"question": query, "answer": answer, "sources": source_text})
        save_data(data)
