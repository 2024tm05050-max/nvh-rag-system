# NVH Compliance RAG System

A Multimodal Retrieval-Augmented Generation (RAG) system for automotive 
NVH (Noise, Vibration & Harshness) compliance documents. Built for the 
BITS Pilani WILP Multimodal RAG Bootcamp.

---

## Problem Statement

**Domain:** Automotive NVH (Noise, Vibration & Harshness) — Design & Development Engineering

**Background**

In automotive NVH development, engineers are responsible for ensuring 
that vehicles meet a complex web of noise-related regulatory requirements 
before they can be homologated and sold in target markets. These requirements 
are codified across multiple international and national standards — including 
UNECE Regulation 51 (exterior noise from motor vehicles), ISO 362 (measurement 
of sound emitted by accelerating road vehicles), IS 3028 Parts 1 & 2 (Indian 
standards for vehicle noise), horn noise regulations, and AVAS (Acoustic Vehicle 
Alerting System) requirements for electric and hybrid vehicles. Each of these 
documents runs to dozens or hundreds of pages and is dense with regulatory tables, 
test procedure descriptions, measurement graphs, and cross-references to other standards.

**The Problem**

During vehicle development, NVH engineers routinely need to answer highly specific, 
configuration-dependent questions: Which exterior noise limit applies to this vehicle 
given its kerb mass, engine type, and category? What are the exact test conditions for 
a hybrid vehicle operating in electric-only mode? Does this powertrain configuration 
trigger AVAS requirements, and if so, what frequency and SPL targets apply? What changed 
between the previous and current revision of ECE R51?

Today, answering these questions requires a combination of manual document reading, 
Ctrl+F keyword searches that fail on scanned content or figures, consulting senior 
colleagues who carry institutional knowledge, and maintaining informal Excel summaries 
that quickly go out of date when standards are revised. This process is slow, error-prone, 
and heavily dependent on individual expertise. A junior engineer or a team working on a 
new market entry may spend hours locating a single applicable limit value — time that 
should be spent on engineering decisions, not document archaeology.

**Why This Problem Is Uniquely Hard**

NVH compliance documents are not amenable to conventional search for several reasons. 
First, the applicability of a limit is almost never a single sentence — it is determined 
by a combination of vehicle category, gross vehicle weight, engine displacement, fuel type, 
and transmission type, typically expressed across multi-column regulatory tables. A keyword 
search for "noise limit" returns dozens of irrelevant rows before reaching the applicable one. 
Second, test conditions for modern powertrains (particularly hybrids and EVs) are described 
procedurally across multiple non-contiguous pages, with conditions that depend on figures and 
graphs that traditional search engines cannot interpret. Third, AVAS requirements are relatively 
new and exist as amendments or annexures embedded within larger regulations, making them easy 
to miss in a linear read. Finally, standard revisions introduce delta changes that are not 
explicitly summarised — engineers must compare versions manually to understand what has changed.

**Why RAG Is the Right Approach**

A retrieval-augmented generation approach is well-suited to this problem for three reasons. 
First, RAG operates over the actual regulatory text without requiring model fine-tuning — the 
source documents remain authoritative and can be updated when standards are revised, without 
retraining. Second, multimodal RAG can process the full document including tables (which encode 
limit applicability logic) and figures (which define test speed profiles and measurement 
conditions) — modalities that keyword search and traditional document Q&A systems ignore 
entirely. Third, RAG produces grounded answers with source references (document name, page 
number, chunk type), which is essential in a compliance context where an engineer must be 
able to cite the exact regulatory clause to a homologation authority.

**Expected Outcomes**

A successful system enables an NVH engineer to ask: *"What is the applicable exterior noise 
limit for an M1 category petrol vehicle?"*, *"What are the AVAS minimum sound requirements 
for a BEV below 20 km/h?"*, or *"What does Table 1 say about load schedule for vehicles 
below 3 tonnes?"* — and receive accurate, cited answers drawn directly from ingested 
regulatory documents.

---

## Architecture Overview

### Ingestion Pipeline
```
PDF Document
    → PyMuPDF Parser
        → Text Chunks (paragraphs)
        → Table Chunks (markdown tables)
        → Image Files (figures/diagrams)
            → GPT-4o-mini Vision → Image Summaries
    → BGE Embedding Model
    → FAISS Vector Store
```

### Query Pipeline
```
User Question
    → BGE Embedding Model
    → FAISS Retriever (Top-K chunks)
    → Context Builder
    → GPT-4o-mini LLM + NVH System Prompt
    → Answer + Source References
```

![Architecture Diagram](screenshots/architecture.png)

---

## Technology Choices

| Component | Choice | Justification |
|---|---|---|
| **Document Parser** | PyMuPDF + pymupdf4llm | Lightweight, no ML model downloads required, excellent table extraction via `find_tables()`, reliable image extraction. Preferred over Docling due to environment compatibility. |
| **Embedding Model** | BGE-small-en-v1.5 (BAAI) | Compact (133MB), strong retrieval performance on technical documents, supports normalised embeddings for cosine similarity. |
| **Vector Store** | FAISS (IndexFlatIP) | In-memory, no server required, inner product search with normalised vectors equals cosine similarity. Sufficient for single-user compliance query workload. |
| **LLM** | GPT-4o-mini | Cost-effective, strong instruction following, low temperature (0.1) for factual regulatory answers. |
| **Vision Model** | GPT-4o-mini | Same model handles both vision and text — reduces API complexity. NVH-specific prompt extracts measurement values and diagram descriptions. |
| **Framework** | FastAPI + direct OpenAI SDK | LangChain abstraction unnecessary for this pipeline complexity. Direct SDK gives full control over prompts and error handling. |

---

## Setup Instructions

### Prerequisites
- Python 3.10 or 3.11
- OpenAI API key with billing enabled
- Git

### 1. Clone the repository
```bash
git clone https://github.com/2024tm05050-max/nvh-rag-system.git
cd nvh-rag-system
```

### 2. Install dependencies
```bash
bash setup.sh
```

### 3. Configure API key
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 4. Pre-build the index (first time only)
```bash
python -c "
from src.ingestion.parser import parse_pdf
from src.models.vision import summarise_all_images
from src.ingestion.embedder import get_embedding_model, embed_chunks, save_index
chunks = parse_pdf('sample_documents/is.3028.1998.pdf')
chunks = summarise_all_images(chunks)
model = get_embedding_model()
embeddings = embed_chunks(chunks, model)
save_index(chunks, embeddings)
print('Index ready!')
"
```

### 5. Start the server
```bash
python main.py
```

### 6. Open Swagger UI
Navigate to: `http://localhost:8000/docs`

---

## API Documentation

### GET /health
Returns system status.

**Response:**
```json
{
  "status": "ok",
  "model_ready": true,
  "indexed_documents": ["is.3028.1998.pdf"],
  "total_chunks": 100,
  "chunk_type_counts": {"text": 45, "table": 7, "image": 48},
  "uptime_seconds": 120.5
}
```

### POST /ingest
Upload a PDF for ingestion.

**Request:** multipart/form-data with `file` field (PDF only)

**Response:**
```json
{
  "message": "Document ingested successfully",
  "filename": "is.3028.1998.pdf",
  "chunks_added": 100,
  "chunk_type_counts": {"text": 45, "table": 7, "image": 48},
  "processing_time_seconds": 45.2
}
```

### POST /query
Query the indexed documents.

**Request:**
```json
{
  "question": "What does Table 1 say about load schedule for vehicles below 3 tonnes?",
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "What does Table 1 say about load schedule?",
  "answer": "Table 1 specifies body weight of 0.12 tonnes for vehicles below 3 tonnes GVW [Source 1]",
  "sources": [
    {
      "filename": "is.3028.1998.pdf",
      "page_number": 8,
      "chunk_type": "table",
      "relevance_score": 0.832,
      "content_preview": "| GVW | Cab Weight | Body Weight |..."
    }
  ],
  "chunks_retrieved": 5
}
```

### GET /documents
List all indexed documents.

### GET /docs
Swagger UI with full API documentation.

---

## Screenshots

### Swagger UI
![Swagger UI](screenshots/swagger_ui.png)

### Health Endpoint
![Health](screenshots/health_endpoint.png)

### Successful Ingestion
![Ingest](screenshots/ingest_response.png)

### Text Query Result
![Text Query](screenshots/query_text.png)

### Table Query Result
![Table Query](screenshots/query_table.png)

### Image Query Result
![Image Query](screenshots/query_image.png)

---

## Limitations & Future Work

## Evaluation

RAGAS automated evaluation was run across 5 NVH-specific test 
questions from IS 3028.

| Metric | Score |
|---|---|
| Faithfulness | 1.000 |
| Answer relevancy | 0.386 |
| Context precision | 0.606 |
| Overall | 0.664 (Good) |

Faithfulness of 1.0 means all answers are fully grounded in 
retrieved source documents — critical for compliance use cases 
where hallucination is unacceptable. Lower answer relevancy 
reflects the system correctly saying "not in context" rather 
than hallucinating an answer. Full evaluation pipeline in 
`src/evaluation/ragas_eval.py`.

**Current Limitations:**
- Ingestion via API times out on Codespaces due to 60s gateway limit — 
  workaround is pre-building the index via terminal script
- FAISS index is rebuilt on each ingest call — does not append to existing index
- Image extraction picks up decorative/logo images in addition to technical figures
- IS 3028 is a measurement method standard; limit values require ingesting 
  additional standards (ECE R51, AIS-049) for complete answers

**Future Work:**
- Add incremental FAISS indexing (append without rebuild)
- Add `/delete` endpoint to remove specific documents
- Implement hybrid search (BM25 + vector) for better keyword matching on 
  regulation numbers
- Add multi-document cross-reference queries (e.g., compare ECE R51.03 vs R51.04)
- Deploy on persistent cloud infrastructure to eliminate timeout issues