# AI101 — An Adaptive Learning System for the Introduction to AI Course

An AI-powered adaptive learning platform that uses LLMs to provide personalised tutoring, generate quizzes and exams, and automatically grade student answers using course materials and marking rubrics.

---

## Project Overview

This platform is fine-tuned on the official curriculum, lecture slides, textbooks, and assessment rubrics of the *Introduction to AI* course. It combines:

- **Conversational tutoring** with RAG-grounded explanations
- **Dynamic quiz/exam generation** with variable difficulty
- **Automated student answer parsing** (free-text and code)
- **Objective evaluation** based on LLM-generated marking grids
- **Student performance analytics** dashboard

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js (App Router, JavaScript) |
| Backend | FastAPI (Python, Uvicorn) |
| Data/ML | Transformers, PEFT, LangChain, FAISS |

---

## Repository Structure

```
AI101/
│
├── data/
│   ├── raw/                       # Original unprocessed course materials
│   ├── preprocessing/             # Data cleaning & transformation
│   └── annotated/                 # Curated datasets for training
│
├── rag_ingestion/                 # Retrieval-Augmented Generation & Ingestion
├── training/                      # LLM fine-tuning
├── grading/                       # Automated grading engine
├── evaluation/                    # System evaluation
│
├── app/
│   ├── frontend/                  # Next.js
│   └── backend/                   # FastAPI
│
├── docs/                          # Documentation
├── notebooks/                     # Jupyter notebooks
├── configs/                       # Global configuration files
├── tests/                         # Unit and integration tests
│
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Module Details

### 1. Data — `data/`

#### `data/raw/`
Original, unprocessed course materials. **Do NOT modify files here** — all transformations happen in `data/preprocessing/`.

| Subdirectory | Contents |
|---|---|
| `Lectures/` | Lecture slides and notes |
| `Textbooks/` | Course textbooks and reference materials |
| `Exams/` | Past midterms and final exams |
| `Tests/` | Past tests and quizzes |
| `Labs/` | Lab materials and exercises |
| `Tutorials/` | Tutorial worksheets and solutions |
| `Resources/` | Additional academic materials |
| `Additional/` | Supplementary materials (syllabi, code repos, marking grids) |

#### `data/preprocessing/`
Scripts and pipelines for cleaning, parsing, chunking, and transforming raw materials.

```
preprocessing/
├── scripts/          # Python scripts for each preprocessing step
└── outputs/          # Processed and cleaned data outputs
```

**Pipeline steps:**
1. **PDF/Slide Parsing** — Extract text, tables, and figures from PDFs and PPTX files.
2. **Text Cleaning** — Remove noise, fix encoding issues, normalise whitespace.
3. **Chunking** — Split documents into semantically meaningful chunks for RAG indexing.
4. **Format Conversion** — Convert to structured JSON/JSONL for downstream tasks.

#### `data/annotated/`
Human-annotated and synthetically generated datasets curated for fine-tuning and evaluation.

```
annotated/
├── qa_pairs/              # Question-answer pairs covering core AI topics
├── rubrics/               # Official marking grids and grading rubrics
└── alignment_dataset/     # Instruction-tuning alignment dataset (prompt-response pairs)
```

- All datasets stored in **JSONL** format.
- QA pairs cover all core AI topics: intelligent agents, search, adversarial search, CSP, knowledge representation, and logic & inference.
- Rubrics map to the official marking grids used in assessments.

---

### 2. RAG & Ingestion — `rag_ingestion/`

End-to-end Retrieval-Augmented Generation (RAG) and document ingestion pipeline for grounding LLM tutoring in actual course materials.

```
rag_ingestion/
├── chunking/        # Chunking strategies (semantic, sliding window, token-based)
├── classifiers/     # Document & content classification
├── embeddings/      # Embedding model wrappers & utilities
├── extractors/      # Text, OCR, table, and figure extractors
├── generation/      # Context builders, prompt builders, and LLM generators
├── loaders/         # Document loaders (PDF, PPTX, DOCX, TXT)
├── retrieval/       # Candidate retrieval & cross-encoder re-ranking
├── storage/         # Vector database (Qdrant) & storage operations
├── tests/           # Unit and integration tests for RAG pipeline
├── config.py        # Pipeline configuration & environment variable loaders
├── pipeline.py      # Document ingestion & indexing orchestration
├── rag_pipeline.py  # End-to-end RAG query & answering pipeline
└── schema.py        # Core data structures and Pydantic schemas
```

- **Loaders & Extractors** — Ingest raw course materials, extract text, tables, figures, and run OCR when needed.
- **Chunking & Classification** — Clean, classify, and split documents into semantically coherent chunks with overlap.
- **Embeddings & Storage** — Compute dense vector embeddings and index them in vector storage (Qdrant).
- **Retrieval & Reranking** — Retrieve candidate chunks and rerank using cross-encoders for high-precision context.
- **Generation** — Token-budgeted context construction and prompt generation for LLM response synthesis.

---

### 3. Training — `training/`

Fine-tuning open-source LLMs on the curated AI course educational corpus.

```
training/
├── configs/         # Training hyperparameter configs (YAML)
├── scripts/         # Training, validation, and export scripts
├── checkpoints/     # Saved model checkpoints (git-ignored)
└── logs/            # Training logs and TensorBoard runs
```

**Training focus areas:**
- Step-by-step reasoning for problem solving
- Mathematical problem solving aligned with course exercises
- Precise alignment with standardised answer keys
- Pedagogical tone appropriate for educational tutoring

> Model checkpoints are **not tracked in git**. Use Weights & Biases or TensorBoard for experiment tracking.

---

### 4. Grading — `grading/`

Automated grading engine that parses student answers and evaluates them against official marking rubrics.

```
grading/
├── rubric_engine/     # Rubric loading, parsing, and matching logic
├── answer_parsing/    # Student answer parsing (free-text and code)
└── scoring/           # Score computation and breakdown generation
```

**Features:**
- Rubric-based grading — maps student responses to rubric criteria.
- Free-text and code parsing.
- Detailed per-criterion score breakdowns with explanations.
- Prevents direct solution leakage during exam mode.
- Complete transparency in marking criteria display.

---

### 5. Evaluation — `evaluation/`

Systematic evaluation of the system's grading accuracy and educational quality.

```
evaluation/
├── metrics/       # Metric computation scripts (correlation, accuracy, etc.)
├── benchmarks/    # Gold-standard benchmark datasets
└── results/       # Evaluation run outputs and reports
```

**Metrics:**
- **Grading accuracy** — Correlation between automated scores and human-provided key answers.
- **Pedagogical explanation clarity** — Quality assessment of generated explanations.
- **Retrieval relevance** — Precision and recall of RAG-retrieved context.
- **Answer alignment** — Degree of match with standardised answer keys.

---

### 6. App — `app/`

The web application layer, split into **Next.js** frontend and **FastAPI** backend.

#### Backend — `app/backend/`

```
backend/
├── main.py            # FastAPI application entry point
├── core/
│   ├── config.py      # Settings (env vars via pydantic-settings)
│   └── dependencies.py
├── api/
│   ├── routes/        # API route handlers
│   │   ├── chat.py    # POST /api/chat — Tutoring chatbot
│   │   ├── quiz.py    # POST /api/quiz/generate — Quiz generation
│   │   ├── grading.py # POST /api/grading/submit — Answer grading
│   │   └── analytics.py # GET /api/analytics/{id} — Performance
│   └── middleware/
├── services/          # Business logic layer
│   ├── chat_service.py
│   ├── quiz_service.py
│   └── grading_service.py
├── models/
│   └── schemas.py     # Pydantic data models
├── utils/
└── requirements.txt
```

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/chat/` | Send message to tutoring chatbot |
| `POST` | `/api/quiz/generate` | Generate a quiz |
| `POST` | `/api/grading/submit` | Submit answer for grading |
| `GET` | `/api/analytics/{student_id}` | Get student performance analytics |

#### Frontend — `app/frontend/`

Next.js application with App Router (JavaScript). Provides:
1. **Conversational Tutoring Chatbot** — Concept explanations grounded in course materials via RAG.
2. **Quiz & Exam Generator** — Automated generation with variable difficulty levels.
3. **Submission Evaluation Interface** — Students submit free-text or code answers for automated grading.
4. **Grading Dashboard** — Detailed score breakdowns against marking rubrics.
5. **Student Performance Analytics** — Track progress and identify weak areas.

---

### 7. Documentation — `docs/`

```
docs/
├── project_report/    # Final project report and deliverables
├── api_docs/          # API reference documentation
└── guides/            # Setup guides, contribution guidelines, user manuals
```

---

### 8. Notebooks — `notebooks/`

```
notebooks/
├── exploration/     # Data exploration and initial analysis
├── experiments/     # Model training experiments and ablation studies
└── demos/           # End-to-end demonstration notebooks
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/AI101.git
cd AI101

# Create a virtual environment & install Python dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the Backend (FastAPI)

```bash
uvicorn app.backend.main:app --reload --port 8000
# API docs → http://localhost:8000/api/docs
```

### Run the Frontend (Next.js)

```bash
cd app/frontend
npm install
npm run dev
# App → http://localhost:3000
```

---

## Core AI Topics Covered

| Topic | Description |
|-------|-------------|
| Intelligent Agents | Agent architectures and environments |
| State-Space Search | BFS, DFS, UCS, A* |
| Adversarial Search | Minimax, Alpha-Beta Pruning |
| CSP | Constraint Satisfaction Problems |
| Knowledge Representation | Propositional & First-Order Logic |
| Logic & Inference | Forward/Backward Chaining, Resolution |

---

## Expected Outcomes

- A fine-tuned open-source educational LLM
- An annotated benchmark dataset of student AI solutions with gold-standard rubrics
- A student performance analytics dashboard
- A fully functional adaptive learning web application

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
