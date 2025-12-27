from langchain_core.tools import tool
from rag import (
    get_rag_chain,
    get_comparison_chain,
    get_comparison_summary_chain,
)

# ==================
# SINGLE PAPER TOOLS
# ==================
def build_single_paper_tool(vectorstore, chat_history: str = ""):
    """
    Tools untuk interaksi dengan SATU dokumen/paper penelitian.
    """
    @tool
    def QA_single_paper(question: str) -> str:
        """
        Gunakan tool ini untuk:
        - pertanyaan spesifik
        - detail faktual
        - siapa / kapan / apa dan sebagainya
        """
        rag_chain = get_rag_chain(
            vectorstore=vectorstore,
            chat_history=chat_history
        )
        return rag_chain.invoke({"question": question})

    @tool
    def summarize_single_paper() -> str:
        """
        Gunakan tool ini jika user meminta:
        - ringkasan
        - overview
        - gambaran umum
        - dokumen ini berisi tentang apa
        """
        rag_chain = get_rag_chain(
            vectorstore=vectorstore,
            chat_history=""
        )

        return rag_chain.invoke({
            "question": "Berikan ringkasan singkat dan terstruktur dari dokumen ini."
        })

    return [QA_single_paper, summarize_single_paper]


# ================
# COMPARISON TOOLS
# ================
def build_comparison_tool(vectorstore_a, vectorstore_b):

    @tool
    def QA_two_papers(question: str) -> str:
        """
        Gunakan tool ini untuk pertanyaan PERBANDINGAN spesifik
        antara dua dokumen/paper.

        Contoh:
        - perbedaan metode
        - perbandingan tujuan
        - perbedaan dataset atau pendekatan
        """
        chain = get_comparison_chain(
            vectorstore_a,
            vectorstore_b
        )
        return chain.invoke({"question": question})

    @tool
    def summarize_two_papers() -> str:
        """
        Gunakan tool ini untuk MERINGKAS dan MEMBANDINGKAN
        dua dokumen/paper secara menyeluruh.
        """
        chain = get_comparison_summary_chain(
            vectorstore_a=vectorstore_a,
            vectorstore_b=vectorstore_b
        )
        return chain.invoke({
            "question": "Ringkas dan bandingkan kedua dokumen"
        })

    return [QA_two_papers, summarize_two_papers]
