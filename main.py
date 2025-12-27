# main.py
import streamlit as st
import uuid
import os
import tempfile
import json

from dotenv import load_dotenv
from ingestion import pdf_to_documents
from qdrant_utils import (
    recreate_collection,
    insert_documents,
    delete_collection,
    get_vectorstore
)
from langchain_core.messages import AIMessage
from supervisor import supervisor_route
from agent_single import build_single_agent
from agent_compare import build_comparison_agent
from agent_chitchat import build_chitchat_agent

load_dotenv()

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="ReAper AI",
    page_icon="📄",
    layout="centered",
)

# ===============================
# Session State Initialization
# ===============================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "papers" not in st.session_state:
    st.session_state.papers = {}

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# 🔑 STEP 2 STATE
if "doc_labels" not in st.session_state:
    st.session_state.doc_labels = {}

if "last_active_doc" not in st.session_state:
    st.session_state.last_active_doc = None

# ===============================
# Sidebar — Upload PDF
# ===============================
st.sidebar.title("📄 Upload Research Paper")

uploaded_files = st.sidebar.file_uploader(
    "Upload maksimal 2 PDF",
    type=["pdf"],
    accept_multiple_files=True,
)

if len(uploaded_files) > 2:
    st.sidebar.warning("Maksimal 2 PDF saja.")
    st.stop()

# ===============================
# Reset Session
# ===============================
if st.sidebar.button("🔄 Reset Session"):
    for paper in st.session_state.papers.values():
        try:
            delete_collection(paper["collection"])
        except Exception:
            pass

    st.session_state.clear()
    st.rerun()

# ===============================
# Incremental PDF Indexing
# ===============================
if uploaded_files:
    with st.spinner("📚 Processing PDFs..."):
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.indexed_files:
                continue

            paper_id = f"paper_{len(st.session_state.papers) + 1}"
            filename = uploaded_file.name
            collection_name = f"session_{st.session_state.session_id}_{paper_id}"

            # Save temp PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            documents = pdf_to_documents(
                pdf_path,
                source_name=uploaded_file.name
            )

            recreate_collection(collection_name)
            insert_documents(collection_name, documents)

            st.session_state.papers[paper_id] = {
                "filename": filename,
                "collection": collection_name,
            }

            # 🔑 STEP 2 — DOC LABELING (INCREMENTAL)
            if len(st.session_state.doc_labels) == 0:
                st.session_state.doc_labels["doc_1"] = collection_name
            elif len(st.session_state.doc_labels) == 1:
                st.session_state.doc_labels["doc_2"] = collection_name

            st.session_state.indexed_files.add(filename)
            os.remove(pdf_path)

    st.session_state.vectorstore_ready = True

# ===============================
# Helper — Chat History
# ===============================
def format_history(messages, k=5):
    return "\n".join(
        f"{m['role']}: {m['content']}"
        for m in messages[-k:]
    )

# ===============================
# Main UI
# ===============================
st.title("📄 ReAper AI")

st.markdown("""
### Research Paper Assistant
""")

if not st.session_state.vectorstore_ready:
    st.info("Silakan upload PDF terlebih dahulu untuk memulai chat.")
    st.stop()

# ===============================
# Render Chat History
# ===============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===============================
# Chat Input
# ===============================
user_query = st.chat_input("Tanyakan isi dokumen...")

if user_query:
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # ===============================
            # SUPERVISOR ROUTING
            # ===============================
            decision = supervisor_route(user_query)

            route = decision.get("route")
            intent = decision.get("intent")
            doc = decision.get("doc")

            if route == "single" and doc == "none":
                if len(st.session_state.doc_labels) == 1:
                    doc = list(st.session_state.doc_labels.keys())[0]

            if doc == "none" and st.session_state.last_active_doc:
                doc = st.session_state.last_active_doc

            if route == "single" and doc not in st.session_state.doc_labels:
                msg = (
                    "Silakan sebutkan dokumen yang dimaksud "
                    "(misalnya: *dokumen pertama* atau *dokumen kedua*)."
                )
                st.markdown(msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": msg
                })
                

            # ===============================
            # BUILD AGENT
            # ===============================
            if route == "single":
                collection = st.session_state.doc_labels[doc]
                vectorstore = get_vectorstore(collection)
                st.session_state.last_active_doc = doc

                agent = build_single_agent(
                    vectorstore=vectorstore,
                    chat_history=format_history(st.session_state.messages)
                )

            elif route == "compare":
                col_a = st.session_state.doc_labels["doc_1"]
                col_b = st.session_state.doc_labels["doc_2"]

                agent = build_comparison_agent(
                    vectorstore_a=get_vectorstore(col_a),
                    vectorstore_b=get_vectorstore(col_b)
                )
            else:
                agent = build_chitchat_agent()

            # ===============================
            # INVOKE AGENT
            # ===============================
            response = agent.invoke({
                "messages": [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
            })

            answer = response["messages"][-1].content
            st.markdown(answer)

            last_msg = response["messages"][-1]

            token_usage = None
            if isinstance(last_msg, AIMessage):
                token_usage = last_msg.response_metadata.get("token_usage")

            with st.sidebar.expander("📊 Token Usage"):
                if token_usage:
                    st.write(f"Input Tokens: {token_usage.get('prompt_tokens')}")
                    st.write(f"Output Tokens: {token_usage.get('completion_tokens')}")
                    st.write(f"Total Tokens: {token_usage.get('total_tokens')}")
                else:
                    st.write(
                        "Token usage tidak tersedia untuk response ini "
                        "(multi-agent / tool-based execution)."
                    )
                        


            # ===============================
            # RAG EVIDENCE
            # ===============================
            if route == "single":
                rag_docs = vectorstore.similarity_search(user_query, k=5)
                if rag_docs:
                    # Ambil source unik
                    sources = {doc.metadata.get("source", "PDF") for doc in rag_docs}

                    with st.expander("📄 Retrieved Documents (RAG Evidence)"):
                        for src in sources:
                            st.markdown(f"- Source: {src}")

            elif route == "compare":
                docs_a = get_vectorstore(col_a).similarity_search(user_query, k=5)
                docs_b = get_vectorstore(col_b).similarity_search(user_query, k=5)

                if docs_a or docs_b:
                    with st.expander("📄 Retrieved Documents (RAG Evidence)"):

                        if docs_a:
                            sources_a = {d.metadata.get("source", "PDF") for d in docs_a}
                            st.markdown("**Dokumen A**")
                            for src in sources_a:
                                st.markdown(f"- Source: {src}")

                        if docs_b:
                            sources_b = {d.metadata.get("source", "PDF") for d in docs_b}
                            st.markdown("**Dokumen B**")
                            for src in sources_b:
                                st.markdown(f"- Source: {src}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
