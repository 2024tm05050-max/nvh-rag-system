"""
NVH Compliance RAG System
Main FastAPI application entry point
"""

from fastapi import FastAPI
from src.api.routes import router
import uvicorn

app = FastAPI(
    title="NVH Compliance RAG System",
    description="Multimodal RAG system for automotive NVH compliance documents",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)