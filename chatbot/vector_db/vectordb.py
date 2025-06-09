import chromadb
from typing import List, Dict, Union
from langchain_ollama import OllamaEmbeddings


class VectorDB:
    def __init__(self, client: chromadb.Client, embedding_model: OllamaEmbeddings):
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

        self.collection.add(
            documents=document,
            metadatas=metadata,
            ids=[str(len(self.collection.get()['ids']) + 1)],
            embeddings=embeddings
        )
