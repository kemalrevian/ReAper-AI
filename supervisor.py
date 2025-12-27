# supervisor.py
from langchain_openai import ChatOpenAI
# from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
import os
import openai
from dotenv import load_dotenv

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
SUPERVISOR_PROMPT = """
Anda adalah Supervisor Agent.

Tugas Anda adalah MENENTUKAN bagaimana sistem harus merespons
pertanyaan user. Anda TIDAK menjawab isi pertanyaan user.

Tentukan:
1. route → "single", "compare", atau "chitchat"
2. intent → "qa", "summary", atau "none"
3. doc → "doc_1", "doc_2", atau "none"

WAJIB mengeluarkan output dalam format JSON PERSIS seperti ini:

{{
  "route": "single | compare | chitchat",
  "intent": "qa | summary | none",
  "doc": "doc_1 | doc_2 | none"s
}}

ATURAN ROUTING:
- Jika user menyapa, mengucapkan terima kasih,
  atau tidak menanyakan isi dokumen → route = "chitchat"
- Jika user membahas SATU dokumen → route = "single"
- Jika user membandingkan DUA dokumen → route = "compare"

ATURAN INTENT:
- Jika user meminta ringkasan / overview → intent = "summary"
- Jika user bertanya detail / spesifik → intent = "qa"
- Jika route = "chitchat" → intent = "none"

ATURAN DOC:
- Jika dokumen disebut eksplisit → doc = "doc_1" atau "doc_2"
- Jika tidak disebut → doc = "none"

PERTANYAAN USER:
{question}
"""

prompt = ChatPromptTemplate.from_template(SUPERVISOR_PROMPT)
chain = prompt | llm


def supervisor_route(question: str) -> dict:
    """
    Menjalankan supervisor routing dan memastikan
    output SELALU berupa JSON valid.
    """
    result = chain.invoke({"question": question})

    try:
        decision = json.loads(result.content)
    except json.JSONDecodeError:
        # Fallback aman agar sistem tidak crash
        decision = {
            "route": "chitchat",
            "intent": "none",
            "doc": "none"
        }

    return decision