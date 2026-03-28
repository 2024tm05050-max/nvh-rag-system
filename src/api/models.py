"""
Pydantic models for API request and response schemas
"""

from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    model_config = {"protected_namespaces": ()}  # add this line
    status: str
    model_ready: bool
    indexed_documents: List[str]
    total_chunks: int
    chunk_type_counts: Dict[str, int]
    uptime_seconds: float


class IngestResponse(BaseModel):
    """Response model for /ingest endpoint"""
    message: str
    filename: str
    chunks_added: int
    chunk_type_counts: Dict[str, int]
    processing_time_seconds: float


class SourceReference(BaseModel):
    """A single source reference in a query response"""
    filename: str
    page_number: int
    chunk_type: str
    relevance_score: float
    content_preview: str


class QueryRequest(BaseModel):
    """Request model for /query endpoint"""
    question: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    """Response model for /query endpoint"""
    question: str
    answer: str
    sources: List[SourceReference]
    chunks_retrieved: int