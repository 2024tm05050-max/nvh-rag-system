"""
PDF Parser module
Extracts text, tables, and images from PDF documents using Docling
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


@dataclass
class ParsedChunk:
    """Represents a single extracted chunk from a PDF"""
    content: str
    chunk_type: str     # "text", "table", or "image"
    page_number: int
    source_file: str
    chunk_index: int


def parse_pdf(pdf_path: str) -> List[ParsedChunk]:
    """
    Parse a PDF file and extract text, tables, and images as chunks.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Parsing PDF: {pdf_path.name}")

    # Download models to local cache if not present
    artifacts_path = StandardPdfPipeline.download_models_hf(force=False)

    # Configure pipeline
    pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True

    # Create converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    # Convert PDF
    print("Running Docling conversion...")
    result = converter.convert(str(pdf_path))
    doc = result.document

    chunks = []
    chunk_index = 0
    filename = pdf_path.name

    # --- Extract TEXT chunks ---
    print("Extracting text chunks...")
    for text_item in doc.texts:
        content = text_item.text.strip()
        if len(content) < 30:
            continue

        page_num = 1
        if text_item.prov:
            page_num = text_item.prov[0].page_no

        chunks.append(ParsedChunk(
            content=content,
            chunk_type="text",
            page_number=page_num,
            source_file=filename,
            chunk_index=chunk_index
        ))
        chunk_index += 1

    # --- Extract TABLE chunks ---
    print("Extracting table chunks...")
    for table_item in doc.tables:
        try:
            table_md = table_item.export_to_markdown(doc=doc)
        except Exception:
            try:
                table_md = table_item.export_to_markdown()
            except Exception:
                table_md = str(table_item)

        if len(table_md.strip()) < 10:
            continue

        page_num = 1
        if table_item.prov:
            page_num = table_item.prov[0].page_no

        chunks.append(ParsedChunk(
            content=table_md,
            chunk_type="table",
            page_number=page_num,
            source_file=filename,
            chunk_index=chunk_index
        ))
        chunk_index += 1

    # --- Extract IMAGE chunks ---
    print("Extracting image chunks...")
    image_dir = Path("data/images") / pdf_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)

    for pic_index, picture_item in enumerate(doc.pictures):
        page_num = 1
        if picture_item.prov:
            page_num = picture_item.prov[0].page_no

        image_path = image_dir / f"page{page_num}_img{pic_index}.png"

        try:
            img = picture_item.get_image(result.document)
            if img is not None:
                img.save(str(image_path), format="PNG")
                chunks.append(ParsedChunk(
                    content=f"[IMAGE: {image_path}]",
                    chunk_type="image",
                    page_number=page_num,
                    source_file=filename,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
            else:
                print(f"  Image {pic_index} returned None, skipping")
        except Exception as e:
            print(f"  Could not save image {pic_index}: {e}")

    print(f"\nParsing complete: {len(chunks)} chunks extracted")
    print(f"  Text:   {sum(1 for c in chunks if c.chunk_type == 'text')}")
    print(f"  Tables: {sum(1 for c in chunks if c.chunk_type == 'table')}")
    print(f"  Images: {sum(1 for c in chunks if c.chunk_type == 'image')}")

    return chunks