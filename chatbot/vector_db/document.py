



class Document:
    def __init__(self, content: str, metadata: dict = None):
        """
        Initializes a Document instance.

        :param content: The text content of the document.
        :param metadata: Optional metadata associated with the document.
        """
        self.content = content
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        return f"Document(content={self.content[:30]}..., metadata={self.metadata})"