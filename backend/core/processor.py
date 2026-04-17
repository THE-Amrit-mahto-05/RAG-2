import fitz  # PyMuPDF
import re
from typing import List, Dict
from backend.api.schema import Chunk

class PDFProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_with_pages(self, pdf_path: str) -> List[Dict]:
        """Extracts text from each page of the PDF."""
        doc = fitz.open(pdf_path)
        pages_content = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            # Clean up text (remove excessive whitespace, etc.)
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                pages_content.append({
                    "page": page_num + 1,
                    "text": text
                })
        doc.close()
        return pages_content

    def create_chunks(self, pages_content: List[Dict]) -> List[Chunk]:
        """Splits extracted text into semantic chunks with page info."""
        all_chunks = []
        chunk_id_counter = 1

        for page in pages_content:
            text = page["text"]
            page_num = page["page"]
            
            # Simple semantic splitting by sentences/paragraphs
            # Here we split by sentences to avoid breaking mid-sentence
            sentences = re.split(r'(?<=[.!?]) +', text)
            
            current_chunk_text = ""
            for sentence in sentences:
                if len(current_chunk_text) + len(sentence) <= self.chunk_size:
                    current_chunk_text += sentence + " "
                else:
                    # Save current chunk
                    if current_chunk_text.strip():
                        all_chunks.append(Chunk(
                            id=f"chunk_{chunk_id_counter:04d}",
                            text=current_chunk_text.strip(),
                            page=page_num,
                            metadata={"char_count": len(current_chunk_text)}
                        ))
                        chunk_id_counter += 1
                    
                    # Start new chunk with overlap
                    # We take the last few sentences as overlap if possible
                    words = current_chunk_text.split()
                    overlap_text = " ".join(words[-20:]) if len(words) > 20 else ""
                    current_chunk_text = overlap_text + " " + sentence + " "

            # Add the last chunk of the page
            if current_chunk_text.strip():
                all_chunks.append(Chunk(
                    id=f"chunk_{chunk_id_counter:04d}",
                    text=current_chunk_text.strip(),
                    page=page_num,
                    metadata={"char_count": len(current_chunk_text)}
                ))
                chunk_id_counter += 1

        return all_chunks

    def process_pdf(self, pdf_path: str) -> List[Chunk]:
        """Complete pipeline: Extract -> Chunk."""
        pages_content = self.extract_text_with_pages(pdf_path)
        chunks = self.create_chunks(pages_content)
        return chunks
