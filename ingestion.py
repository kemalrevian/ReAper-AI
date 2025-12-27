from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text

def chunking_long_text(content):
    splitter = RecursiveCharacterTextSplitter(
        # separators=[".", " "],
        chunk_size = 1000,
        chunk_overlap=10
    )
    chunks = splitter.split_text(content)
    return chunks


def chunks_to_documents(chunks, source):
    documents = []

    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": source,         
                "chunk_id": i              
            }
        )
        documents.append(doc)

    return documents


def pdf_to_documents(file_path: str, source_name: str) -> list[Document]:
    """
    Full ingestion pipeline:
    PDF -> text -> chunks -> Documents
    """
    text = extract_text_from_pdf(file_path)
    chunks = chunking_long_text(text)

    documents = chunks_to_documents(chunks, source_name)
    return documents