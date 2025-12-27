from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

import os
import openai
from dotenv import load_dotenv

# =====
# MODEL
# =====
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = "gpt-4o-mini"

llm = ChatOpenAI(model=CHAT_MODEL,temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

single_prompt = """
Anda adalah AI Research Assistant.

ATURAN KERAS (WAJIB DIPATUHI):
1. Jawab HANYA berdasarkan informasi yang terdapat
   dalam bagian "KONTEKS".
2. DILARANG menggunakan pengetahuan umum,
   asumsi, atau informasi di luar konteks.
3. Anda BOLEH:
   - merangkum
   - menyusun ulang
   - menjelaskan dengan bahasa sendiri
   SELAMA maknanya SETARA dengan teks sumber.
4. Jika informasi yang ditanyakan TIDAK tersedia
   secara eksplisit dalam konteks, jawab:
   "Informasi tidak tersedia dalam dokumen."
5. Jangan menambahkan opini, interpretasi baru,
   atau spekulasi.

KONTEKS:
{context}

RIWAYAT PERCAKAPAN (untuk referensi, bukan sumber fakta):
{chat_history}

PERTANYAAN:
{question}

JAWABAN:
- Singkat
- Faktual
- Terstruktur jika memungkinkan
"""


prompt = ChatPromptTemplate.from_template(single_prompt)

def format_history(messages):
    return "\n".join(
        f"{m['role']}: {m['content']}"
        for m in messages[-5:]
    )
def get_rag_chain(vectorstore, chat_history: str = ""):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    rag_chain = (
        {
            "context": itemgetter("question")
                       | retriever
                       | RunnableLambda(format_docs),
            "question": itemgetter("question"),
            "chat_history": lambda _: chat_history,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


comparison_qa_prompt = """
Anda adalah AI Research Assistant.

Tugas Anda adalah MEMBANDINGKAN dua dokumen
berdasarkan informasi yang tersedia.

ATURAN KERAS:
1. Gunakan HANYA informasi dari konteks Paper A dan Paper B.
2. DILARANG menggunakan pengetahuan eksternal.
3. Jangan mengarang atau menyimpulkan di luar teks.
4. Jika informasi perbandingan tidak tersedia secara eksplisit,
   jawab:
   "Informasi tidak tersedia dalam dokumen."

KONTEKS PAPER A:
{context_a}

KONTEKS PAPER B:
{context_b}

PERTANYAAN:
{question}

JAWABAN:
- Fokus pada perbandingan langsung
- Jawaban ringkas dan faktual
"""
comparison_qa = ChatPromptTemplate.from_template(
    comparison_qa_prompt
)


def get_comparison_chain(vectorstore_a, vectorstore_b):
    retriever_a = vectorstore_a.as_retriever(
        search_kwargs={"k": 10}
    )
    retriever_b = vectorstore_b.as_retriever(
        search_kwargs={"k": 10}
    )

    chain = (
        {
            "context_a": itemgetter("question") | retriever_a | RunnableLambda(format_docs),
            "context_b": itemgetter("question") | retriever_b | RunnableLambda(format_docs),
            "question": itemgetter("question"),
        }
        | comparison_qa
        | llm
        | StrOutputParser()
    )

    return chain


COMPARISON_SUMMARY_PROMPT = """
Anda adalah AI Research Assistant.

Tugas Anda adalah MERINGKAS dan MEMBANDINGKAN
dua dokumen secara menyeluruh.

ATURAN KERAS:
- Gunakan HANYA informasi dari konteks
- DILARANG menambahkan pengetahuan luar
- DILARANG menyimpulkan di luar teks
- Jika informasi tidak tersedia, nyatakan dengan jelas

FORMAT WAJIB:

### Ringkasan Dokumen A
- Fokus utama:
- Topik penting:

### Ringkasan Dokumen B
- Fokus utama:
- Topik penting:

### Perbandingan
- Persamaan:
- Perbedaan:

KONTEKS DOKUMEN A:
{context_a}

KONTEKS DOKUMEN B:
{context_b}
"""

comparison_summary = ChatPromptTemplate.from_template(COMPARISON_SUMMARY_PROMPT)

def get_comparison_summary_chain(vectorstore_a, vectorstore_b):
    retriever_a = vectorstore_a.as_retriever(search_kwargs={"k": 10})
    retriever_b = vectorstore_b.as_retriever(search_kwargs={"k": 10})

    chain = (
        {
            "context_a": itemgetter("question") | retriever_a | RunnableLambda(format_docs),
            "context_b": itemgetter("question") | retriever_b | RunnableLambda(format_docs),
        }
        | comparison_summary
        | llm
        | StrOutputParser()
    )

    return chain