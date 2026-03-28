"""
Embedding and Vector Store module
Converts chunks into vectors and stores them in FAISS
"""

import os
import json
import pickle
from pathlib import Path
from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.ingestion.parser import ParsedChunk


# Path where FAISS index and metadata are saved
INDEX_PATH = Path("data/faiss_index")


def get_embedding_model():
    """Load the BGE embedding model"""
    print("Loading embedding model (BGE)...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return model


def embed_chunks(chunks: List[ParsedChunk], model: SentenceTransformer) -> np.ndarray:
    """
    Convert chunks to embedding vectors.
    
    Args:
        chunks: List of ParsedChunk objects
        model: SentenceTransformer model
        
    Returns:
        numpy array of embeddings
    """
    print(f"Embedding {len(chunks)} chunks...")
    
    # Prepare texts for embedding
    # BGE works best with a prefix for retrieval
    texts = [f"Represent this NVH document chunk: {chunk.content}" 
             for chunk in chunks]
    
    # Generate embeddings in batches
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True  # normalize for cosine similarity
    )
    
    return embeddings


def save_index(chunks: List[ParsedChunk], embeddings: np.ndarray):
    """
    Save FAISS index — appends to existing index if present.
    New documents are added without removing previously indexed ones.
    """
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    
    index_file = INDEX_PATH / "index.faiss"
    metadata_file = INDEX_PATH / "metadata.json"
    
    # Load existing index if present
    if index_file.exists() and metadata_file.exists():
        existing_index = faiss.read_index(str(index_file))
        with open(metadata_file, "r") as f:
            existing_metadata = json.load(f)
        
        # Check for duplicate document — remove old chunks first
        new_source = chunks[0].source_file if chunks else None
        existing_metadata = [
            m for m in existing_metadata 
            if m["source_file"] != new_source
        ]
        
        # Rebuild index without old chunks from this document
        if len(existing_metadata) > 0:
            kept_indices = [m["index"] for m in existing_metadata]
            kept_embeddings = np.array([
                existing_index.reconstruct(int(i)) 
                for i in range(existing_index.ntotal)
                if i in set(kept_indices)
            ], dtype=np.float32)
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            if len(kept_embeddings) > 0:
                index.add(kept_embeddings)
            
            # Re-number metadata indices
            for i, m in enumerate(existing_metadata):
                m["index"] = i
        else:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
    else:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        existing_metadata = []
    
    # Add new chunks
    start_idx = index.ntotal
    index.add(embeddings.astype(np.float32))
    
    # Build new metadata entries
    new_metadata = []
    for i, chunk in enumerate(chunks):
        new_metadata.append({
            "index": start_idx + i,
            "content": chunk.content,
            "chunk_type": chunk.chunk_type,
            "page_number": chunk.page_number,
            "source_file": chunk.source_file,
            "chunk_index": chunk.chunk_index
        })
    
    # Save combined index and metadata
    all_metadata = existing_metadata + new_metadata
    faiss.write_index(index, str(index_file))
    
    with open(INDEX_PATH / "metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"Index updated: {index.ntotal} total vectors")
    print(f"  Added: {len(chunks)} new chunks from {chunks[0].source_file}")
    print(f"  Total documents: {len(set(m['source_file'] for m in all_metadata))}")


def load_index():
    """
    Load FAISS index and metadata from disk.
    
    Returns:
        tuple: (faiss_index, metadata_list) or (None, None) if not found
    """
    index_file = INDEX_PATH / "index.faiss"
    metadata_file = INDEX_PATH / "metadata.json"
    
    if not index_file.exists() or not metadata_file.exists():
        return None, None
    
    index = faiss.read_index(str(index_file))
    
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    return index, metadata


def get_index_stats():
    """Return stats about the current index"""
    index, metadata = load_index()
    
    if index is None:
        return {
            "total_chunks": 0,
            "indexed_documents": [],
            "chunk_type_counts": {}
        }
    
    # Count by type
    type_counts = {}
    documents = set()
    for item in metadata:
        chunk_type = item["chunk_type"]
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        documents.add(item["source_file"])
    
    return {
        "total_chunks": index.ntotal,
        "indexed_documents": list(documents),
        "chunk_type_counts": type_counts
    }