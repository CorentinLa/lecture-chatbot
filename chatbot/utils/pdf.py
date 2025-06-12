import fitz
from fitz import Page
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFReader:
    def __init__(self, file, user: int, doc_name: str = None):
        self.user = user

        try:
            self.doc = fitz.open(stream=file, filetype="pdf")
        except Exception as e:
            raise ValueError(f"Could not open PDF file. Error: {e}")
        
        self.pages = {page.number: self.parse_blocks(page) for page in self.doc}

        self.title = self.doc.metadata["title"] if self.doc.metadata["title"] != "" else "Untitled"
        self.name = doc_name
        self.author = self.doc.metadata["author"] if self.doc.metadata["author"] != "" else "Unknown"


    def parse_blocks(self, page: Page):
        text_dict = page.get_text("dict")
        blocks = []
        for block in text_dict.get("blocks", []):
            if block == []:
                continue
            if block["type"] == 0: # 0 for text blocks (https://pymupdf.readthedocs.io/en/latest/textpage.html#TextPage.extractBLOCKS)
                block_text = []
                for line in block.get("lines", []):
                    block_text.append(" ".join(span["text"] for span in line.get("spans", [])))

                blocks.append("\n".join(block_text))
        
        return blocks
    
    def chunk(self,
        chunk_size_blocks: int = 3,
        chunk_overlap_blocks: int = 1,
        chunk_size_chars: int = 512,
        chunk_overlap_chars: int = 100
    ) -> list[Document]:
        """
        Divides the PDF content into chunks based on the specified parameters.
        Args:
            chunk_size_blocks (int): Number of blocks per pre-chunk.
            chunk_overlap_blocks (int): Number of overlapping blocks between pre-chunks.
            chunk_size_chars (int): Maximum number of characters per chunk.
            chunk_overlap_chars (int): Number of overlapping characters between chunks.
        """
        base_chunks = []

        for page_number, blocks in self.pages.items():
            num_blocks = len(blocks)
            start_idx = 0

            while start_idx < num_blocks:
                end_idx = min(start_idx + chunk_size_blocks, num_blocks)
                window_blocks = blocks[start_idx:end_idx]
                content = "\n\n".join(window_blocks)

                base_chunks.append(Document(
                    page_content=content,
                    metadata={
                        "page": page_number + 1,
                        "source": self.name,
                        "title": self.title,
                        "author": self.author,
                        "user": self.user
                    }
                ))

                if end_idx == num_blocks:
                    break
                start_idx += chunk_size_blocks - chunk_overlap_blocks

        # Maintenant on utilise le splitter LangChain
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_chars,
            chunk_overlap=chunk_overlap_chars,
        )
        self.final_chunks = text_splitter.split_documents(base_chunks)


        return self.final_chunks
        

if __name__ == "__main__":
    pdf_reader = PDFReader("example.pdf", user=1)

    pdf_reader.chunk(
        chunk_size_blocks=3,
        chunk_overlap_blocks=1,
        chunk_size_chars=512,
        chunk_overlap_chars=100
    )

    print(pdf_reader.final_chunks[15])
