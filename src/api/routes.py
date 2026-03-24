"""
FastAPI route definitions
All API endpoints are defined here
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok", "message": "NVH RAG system is running"}