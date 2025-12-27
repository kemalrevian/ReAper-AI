from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import openai
from dotenv import load_dotenv
import os
from tools import build_comparison_tool

# =====
# MODEL
# =====
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = "gpt-4o-mini"

llm = ChatOpenAI(model=CHAT_MODEL,temperature=0)

# =============
# SYSTEM PROMPT
# =============
SYSTEM_PROMPT = """
Anda adalah Comparison Research Agent.

Tugas Anda adalah membandingkan DUA dokumen penelitian
menggunakan tools yang tersedia.

ATURAN PENTING:
1. Gunakan tool "summarize_two_papers" jika user meminta:
   - ringkasan perbandingan
   - gambaran umum kedua dokumen
   - persamaan dan perbedaan secara umum
2. Gunakan tool "QA_two_papers" untuk pertanyaan perbandingan SPESIFIK,
   seperti:
   - perbedaan metode
   - perbedaan tujuan
   - perbandingan algoritma atau pendekatan
3. Jangan menggunakan pengetahuan eksternal.
4. Jangan menjawab langsung tanpa tool.
5. Jika informasi perbandingan tidak tersedia,
   jawab:
   "Informasi tidak tersedia dalam dokumen."

Fokuskan jawaban pada perbandingan yang diminta user.
"""

def build_comparison_agent(vectorstore_a, vectorstore_b):
    tools = build_comparison_tool(
        vectorstore_a=vectorstore_a,
        vectorstore_b=vectorstore_b
    )

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT
    )
