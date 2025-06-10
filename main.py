from typing import Union

from fastapi import FastAPI

import sys
import os
sys.path.append("./lecture-chatbot")

from chromadb import PersistentClient

from chatbot.models.chat_llm import ChatbotLLM
from chatbot.models.embedding import EmbeddingModel
from chatbot.vector_db.vectordb import VectorDB


app = FastAPI()

# Initialize the chatbot with the Mistral model running on Ollama server
chatbot = ChatbotLLM(
    model_name="mistral:latest",
    temperature=0.8,
    num_predict=512,
    base_url="http://127.0.0.1:11434/",
)

embedding_model = EmbeddingModel(
    model_name="nomic-embed-text:latest",
    base_url="http://127.0.0.1:11434/",
)

if not os.path.exists("./lecture-chatbot/chatbot/data/vector_db/chroma_db"):
    os.makedirs("./lecture-chatbot/chatbot/data/vector_db/chroma_db")

db_client = PersistentClient(
    path="./lecture-chatbot/chatbot/data/vector_db/chroma_db",
)

rag_db = VectorDB(
    client=db_client,
    embedding_model=embedding_model,
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.put("/chat/infer")
def chat_infer(prompt: str):
    return chatbot.invoke(message=prompt)


@app.put("/debug/embedding")
def debug_embedding(text: Union[str, list[str]]):
    """
    Endpoint to test the embedding model.
    Accepts a single string or a list of strings and returns their embeddings.
    """
    if isinstance(text, str):
        text = [text]

    embeddings = embedding_model.embed_documents(text)
    return {"embeddings": embeddings}


@app.put("/debug/insert_document")
def insert_document(
    document: Union[str, list[str]] = "Document content",
    metadata: Union[dict[str, str], list[dict[str, str]]] = {
        "source": "unknown",
        "author": "unknown",
    },
):
    """
    Endpoint to insert a document into the vector database.
    Accepts a single string or a list of strings as the document and corresponding metadata.
    """
    rag_db.add_document(document=document, metadata=metadata)
    return {"status": "Document(s) added successfully"}


@app.get("/debug/retrieve_documents")
def retrieve_documents(query: str, limit: int = 5):
    """
    Endpoint to retrieve documents from the vector database based on a query.
    Returns a list of documents and their metadata.
    """
    query = embedding_model.embed_query(query)

    results = rag_db.collection.query(
        query_embeddings=[query],
        n_results=limit,
    )

    return {
        "documents": results["documents"],
        "metadatas": results["metadatas"],
        "distances": results["distances"],
    }
