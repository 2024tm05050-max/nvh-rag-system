from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
import uvicorn

app = FastAPI(
    title="NVH Compliance RAG System",
    description="Multimodal RAG system for automotive NVH compliance documents",
    version="1.0.0"
)

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)