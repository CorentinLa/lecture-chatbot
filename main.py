from typing import Union, Annotated

from fastapi import FastAPI, File, UploadFile

import sys
import os

sys.path.append("./lecture-chatbot")

from chromadb import PersistentClient

from chatbot.models.chat_llm import ChatbotLLM
from chatbot.models.embedding import EmbeddingModel
from chatbot.vector_db.vectordb import VectorDB
from chatbot.utils.pdf import PDFReader


app = FastAPI()

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

chatbot = ChatbotLLM(
    model_name="mistral:latest",
    temperature=0.8,
    num_predict=512,
    base_url="http://127.0.0.1:11434/",
    embedding_model=embedding_model,
    rag_db=rag_db,
)




@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.put("/chat/infer")
def chat_infer(prompt: str):
    return chatbot.invoke(message=prompt, user=1)


@app.put("/debug/reset_vector_db")
def reset_vector_db():
    """
    Endpoint to reset the vector database.
    Deletes all documents and metadata from the vector database.
    """
    rag_db.reset()
    return {"status": "Vector database reset successfully"}

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


@app.post("/debug/upload_pdf")
def upload_pdf(file: Annotated[bytes, File(description="Upload a PDF file")]):
    """
    Endpoint to upload a PDF file and process it
    """
    try:
        pdf_reader = PDFReader(file, user=1)
        pdf_reader.chunk(
            chunk_size_blocks=3,
            chunk_overlap_blocks=1,
            chunk_size_chars=512,
            chunk_overlap_chars=100,
        )
    except Exception as e:
        return {"error": f"Failed to process PDF: {str(e)}"}

    return {"status": "PDF uploaded and processed successfully",
            "chunks": pdf_reader.final_chunks
            }


@app.post("/file/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and process it into the vector_db.
    """
    try:
        contents = await file.read()
        filename = file.filename
        pdf_reader = PDFReader(contents, user=1, doc_name=filename)
        pdf_reader.chunk(
            chunk_size_blocks=3,
            chunk_overlap_blocks=1,
            chunk_size_chars=1500,
            chunk_overlap_chars=200,
        )

        # Add the processed chunks to the vector database
        rag_db.add_langchain_documents(pdf_reader.final_chunks)

    except Exception as e:
        return {"error": f"Failed to process PDF: {str(e)}"}

    return {"status": "PDF uploaded and processed successfully"}
