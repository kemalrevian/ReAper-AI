from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import openai
from dotenv import load_dotenv
import os
from tools import build_single_paper_tool

# ===============================
# MODEL
# ===============================
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = "gpt-4o-mini"

llm = ChatOpenAI(model=CHAT_MODEL,temperature=0)

# ===============================
# SYSTEM PROMPT
# ===============================
SINGLE_SYSTEM_PROMPT = """
Anda adalah Single Document Research Agent.

Tugas Anda adalah menjawab pertanyaan user
berdasarkan SATU dokumen penelitian menggunakan tools yang tersedia.

ATURAN PENTING:
1. Gunakan tool "summarize" untuk pertanyaan bersifat UMUM atau GLOBAL,
   seperti:
   - "paper ini membahas apa"
   - "isi dokumen ini tentang apa"
   - "apa tujuan penelitian ini"
2. Gunakan tool "qa" untuk pertanyaan SPESIFIK atau DETAIL,
   seperti:
   - algoritma yang digunakan
   - metode penelitian
   - dataset, variabel, atau hasil tertentu
3. Jangan menjawab langsung tanpa tool.
4. Jangan menggunakan pengetahuan di luar dokumen.
5. Jika informasi tidak tersedia dalam dokumen,
   jawab:
   "Informasi tidak tersedia dalam dokumen."

Tujuan Anda adalah memberikan jawaban yang
ringkas, relevan, dan sesuai dengan jenis pertanyaan.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SINGLE_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
])

def build_single_agent(vectorstore, chat_history=""):
    tools = build_single_paper_tool(
        vectorstore=vectorstore,
        chat_history=chat_history
    )

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt
    )
