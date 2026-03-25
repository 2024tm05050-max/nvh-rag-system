"""
Quick test for the PDF parser
Run: python test_parser.py
"""

from src.ingestion.parser import parse_pdf

# Replace with your actual PDF filename
PDF_PATH = "sample_documents/is.3028.1998.pdf"

chunks = parse_pdf(PDF_PATH)

print("\n--- SAMPLE CHUNKS ---")
for chunk in chunks[:5]:
    print(f"\nType: {chunk.chunk_type}")
    print(f"Page: {chunk.page_number}")
    print(f"Content preview: {chunk.content[:200]}")
    print("-" * 40)