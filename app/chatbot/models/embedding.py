from langchain_ollama import OllamaEmbeddings

from typing import List

class EmbeddingModel:
    """
    OllamaEmbeddings based class to add potentially more functionality in the future.
    """
    def __init__(self, model_name: str = "nomic-embed-text:latest", base_url: str = "http://127.0.0.1:11434/"):
        """
        Initializes the embedding model with the specified model name and base URL.

        :param model_name: The name of the embedding model to use.
        :param base_url: The base URL for the Ollama server.
        """
        self.embedding_model = OllamaEmbeddings(
            model=model_name,
            base_url=base_url,
        )

    def embed_query(self, text: str) -> list:
        """
        Embeds a single text query.

        :param text: The text to embed.
        :return: The embedding vector as a list.
        """
        return self.embedding_model.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> list:
        """
        Embeds multiple text documents.

        :param texts: A list of texts to embed.
        :return: A list of embedding vectors.
        """
        return self.embedding_model.embed_documents(texts)