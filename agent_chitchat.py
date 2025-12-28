from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import openai
from dotenv import load_dotenv
import os

# =====
# MODEL
# =====
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = "gpt-4o-mini"

llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)


# =============
# SYSTEM PROMPT
# =============
CHITCHAT_PROMPT = """
Anda adalah Chitchat Agent untuk ReAper AI.

ATURAN KERAS:
1. Anda HANYA boleh:
   - Menyapa pengguna
   - Menjelaskan kemampuan ReAper AI
   - Menanggapi ucapan terima kasih / penutup
2. DILARANG:
   - Menjelaskan konsep teknis
   - Memberikan definisi, teori, atau pengetahuan umum
3. Jika user bertanya tentang konsep atau istilah:
   Jawab dengan sopan:
   "Pertanyaan tersebut tidak dijelaskan secara eksplisit dalam dokumen yang diunggah."
4. Jangan menggunakan pengetahuan di luar dokumen.

Jawaban singkat dan ramah.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", CHITCHAT_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
])

def build_chitchat_agent():
    """
    Agent khusus untuk menangani chitchat / small talk.
    Tidak menggunakan tools dan tidak terhubung ke RAG.
    """
    return create_react_agent(
        model=llm,
        tools=[],
        prompt=prompt
    )