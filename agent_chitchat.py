from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

CHITCHAT_PROMPT = """
Anda adalah AI Research Assistant.

Tugas Anda:
- Menyapa user dengan sopan dan ramah
- Menanggapi ucapan terima kasih atau percakapan ringan
- Mengarahkan user untuk bertanya tentang isi dokumen PDF

ATURAN KERAS:
- JANGAN menjawab isi dokumen
- JANGAN menggunakan pengetahuan eksternal
- Jawaban harus singkat, natural, dan membantu
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
        tools=[],   # tidak ada tool
        prompt=prompt
    )