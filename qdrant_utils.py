from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
import os
import openai
from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

#Create/Reset collection
def recreate_collection(collection_name: str):
    """
    Delete collection if exists, then create a fresh one.
    """
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1536,               
            distance=Distance.COSINE
        )
    )

# Insert documents into collection
def insert_documents(collection_name: str, documents: list):
    """
    Insert documents into an existing Qdrant collection.
    """
    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

# Retrieve vectorstore for querying
def get_vectorstore(collection_name: str) -> QdrantVectorStore:
    """
    Connect to an existing collection for retrieval.
    """
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

# Delete collection
def delete_collection(collection_name: str):
    """
    Permanently delete a collection.
    """
    client.delete_collection(collection_name)
