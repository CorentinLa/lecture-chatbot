import chromadb
from typing import List, Dict, Union
from chatbot.models.embedding import EmbeddingModel
from langchain_core.documents import Document


class VectorDB:
    def __init__(self, client: chromadb.Client, embedding_model: EmbeddingModel):
        self.client = client
        self.collection = client.get_or_create_collection(name="documents")
        self.embedding_model = embedding_model

    def add_document(self, document: Union[str, List[str]], metadata: Union[Dict, List[Dict]]):
        """Adds a document to the vector database with associated metadata.
        Args:
            document (Union[str, List[str]]): The document or list of documents to add.
            metadata (Union[Dict, List[Dict]]): Metadata associated with the document(s).
        """
        if isinstance(document, str):
            document = [document]
        if isinstance(metadata, dict):
            metadata = [metadata]

        if len(document) != len(metadata):
            raise ValueError("The number of documents must match the number of metadata entries.")
        
        # TODO: Eventually use async embedding, or implement chromaDB embedding function to use async embedding https://docs.trychroma.com/docs/run-chroma/python-http-client
        embeddings = [self.embedding_model.embed_query(doc) for doc in document]

        current_id = len(self.collection.get()['ids']) if self.collection.get()['ids'] else 0
        ids = [str(current_id + i) for i in range(len(document))]

        self.collection.add(
            documents=document,
            metadatas=metadata,
            ids=ids,
            embeddings=embeddings
        )

    def add_langchain_documents(self, documents: List[Document]):
        """Adds a LangChain Documents to the vector database.
        Args:
            documents (Document): The documents to add.
        """
        if isinstance(documents, Document):
            documents = [documents]

        docs = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        self.add_document(
            document=docs,
            metadata=metadatas
        )

    def reset(self):
        """Resets the vector database by deleting all documents and metadata."""
        self.client.delete_collection(name="documents")
        self.collection = self.client.get_or_create_collection(name="documents")
