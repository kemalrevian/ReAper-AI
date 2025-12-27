# ReAper AI — Research Paper Assistant

ReAper AI is a chatbot designed to help users **analyze, summarize, and compare research papers** using **PDF documents uploaded by the user**.

This application was developed to showcase the implementation of **Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) pipelines, and multi-agent architectures** using LangChain and LangGraph.

---

## 🚀 Key Features

- 📄 **Document-Grounded Question Answering**
  - Answers are generated **strictly from uploaded PDF documents**
  - Explicitly refuses to answer if information is not found in the documents

- 🧠 **Multi-Agent Architecture**
  - **Supervisor Agent**: routes user intent (single paper, comparison, or chitchat)
  - **Single Paper Agent**: QA and summarization for one document
  - **Comparison Agent**: compares two research papers
  - **Chitchat Agent**: handles non-document-related conversation

- 🔎 **Retrieval-Augmented Generation (RAG)**
  - PDF → chunking → embedding → Qdrant Vector Database
  - LLM responses are grounded on retrieved document context

- 💬 **Chat History Awareness**
  - Maintains conversational context across multiple turns

- 📊 **Token Usage Monitoring**
  - Displays input, output, and total token usage per response

- 📄 **RAG Evidence Display**
  - Shows document sources used to generate each answer

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** — UI and cloud deployment
- **LangChain & LangGraph** — agent orchestration
- **OpenAI (GPT-4o-mini)** — Large Language Model
- **Qdrant Cloud** — Vector Database
- **PyPDF** — PDF text extraction

