import os
from backend.core.processor import PDFProcessor

def test_pdf_processing():
    # Note: This requires a sample PDF. 
    # For testing purposes, we can see if the import and initialization works.
    processor = PDFProcessor()
    print("PDFProcessor initialized successfully.")
    
    # We'll need a real PDF to test extraction.
    # I'll create a dummy check.
    if hasattr(processor, 'process_pdf'):
        print("process_pdf method exists.")

if __name__ == "__main__":
    test_pdf_processing()
