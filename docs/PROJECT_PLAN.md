# AI Studio — Project Plan

## Project

AI Prompt Playground → AI Studio

## Final Goal

Build a production-style AI platform combining:

- LLMs
- RAG
- Vector database
- Tools
- Agents
- MCP
- Observability
- Evals

The project will be developed progressively rather than building the entire
final architecture upfront.

---

# Current Progress

## Completed

- Python
- FastAPI
- AI Module 1
- AI Module 2
- Sprint 1 — Project Setup
- Sprint 2 — Prompt Builder
- Sprint 3 — LLM Integration
- Sprint 4 — Prompt Experiments + Basic Evals
- Sprint 5 — Structured AI
- Sprint 6 — Prompt Library

## Current

Sprint 7 — RAG + RAG Evals

## Next

Sprint 8 — Search / Retrieval Evaluation

---

# 12-Sprint Roadmap

## Sprint 1 — Project Setup

AI Module: Module 2

Focus:

- React
- Vite
- TypeScript
- FastAPI project setup
- Uvicorn
- Project structure
- GET / POST
- Pydantic request/response models
- CORS
- Environment variables
- API service layer
- Health endpoint
- Mock chat API
- React ↔ FastAPI communication

Project features:

- React UI
- FastAPI backend
- `/health`
- Mock `/chat`
- Basic API service layer

---

## Sprint 2 — Prompt Builder

AI Module: Module 2

Focus:

- Prompt structure
- System instructions
- User instructions
- Examples
- Context

Project features:

- Prompt builder UI
- Prompt construction API
- Structured prompt representation

---

## Sprint 3 — LLM Integration

Module 3 begins.

Status: Complete

Focus:

- Connect the application to an actual LLM
- Establish the model abstraction
- Replace mock AI responses with real model responses
- Handle model configuration
- Handle AI API errors
- Establish a clean LLM service layer

Delivered:

- Gemini and Groq providers behind an LLM provider abstraction
- Environment-based provider and model configuration
- Non-streaming `POST /chat` model responses
- Provider, service, and HTTP plain-text streaming through `POST /chat/stream`
- Provider error normalization and basic tests

---

## Sprint 4 — Prompt Experiments + Basic Evals

Status: Complete

Focus:

- Prompt experimentation
- Comparing outputs
- Initial evaluation concepts
- Expected output vs actual output
- Basic evaluation workflow

Delivered:

- Zero-shot, one-shot, and few-shot prompt experiments
- Same-input prompt/output comparison across configurations
- Evaluation datasets with input, expected output, and criteria
- Dataset-driven evaluation runs through the existing chat service
- Expected versus actual output results with deterministic pass/fail checks

---

## Sprint 5 — Structured AI

Status: Complete

Focus:

- Structured outputs
- Reliable response formats
- Backend validation
- Application-level handling of structured AI responses

Delivered:

- Strict Pydantic-validated structured chat responses
- JSON, Markdown, Table, Email, and HTML formatted-output API
- Provider-neutral tool-calling contract and backend orchestration demo
- Provider-neutral Base64 image generation API
- Provider-neutral Base64 audio/TTS generation API
- Gemini and Groq Sprint 5 provider capability implementations and tests

---

## Sprint 6 — Prompt Library

Status: Complete

Focus:

- Saving prompts
- Organizing prompts
- Reusing prompts
- Prompt management
- Persistence

Delivered:

- Neon PostgreSQL configuration through `DATABASE_URL`
- SQLAlchemy/psycopg lazy engine and request-session foundation
- Shared declarative base for future persistence models
- Alembic setup using the shared database URL and `Base.metadata`, ready for
  the first model-backed migration
- Core `Prompt` SQLAlchemy model with title, content, metadata flags, and
  timestamps
- Applied initial `prompts` table migration on Neon PostgreSQL
- Prompt repository/service layers with schema-based create, update, and read
  boundaries
- Prompt Library CRUD endpoints using request-scoped sessions
- Favorites, simple categories, and reusable templates on the existing Prompt
  entity, with no additional tables
- PostgreSQL-backed Prompt Library search, filters, and offset-limit pagination
- Immutable content-only `PromptVersion` history and version-history APIs:
  V1 at prompt creation and subsequent versions only for actual content
  changes; prompt metadata remains current state
- Safe Version 1 backfill for existing Neon prompts

---

## Sprint 7 — RAG + RAG Evals

Focus:

- RAG
- Document ingestion
- Embeddings
- Vector database
- Retrieval
- RAG evaluation

---

## Sprint 8 — Search / Retrieval Evaluation

Focus:

- Retrieval quality
- Search evaluation
- Improving retrieved context
- Measuring retrieval behaviour

---

## Sprint 9 — Agents + MCP

Focus:

- Agents
- Tools
- MCP
- Tool execution
- Agent workflows

---

## Sprint 10 — Multi-Agent + Advanced Evals

Focus:

- Multi-agent workflows
- Advanced evaluation
- Trajectory evaluation
- Agent evaluation

---

## Sprint 11 — Production + Full Evaluation System

Focus:

- Production architecture
- Full evaluation system
- Observability
- Reliability
- Error handling
- Performance

---

## Sprint 12 — Portfolio + Regression Dashboard

Focus:

- Portfolio polish
- Regression testing
- Evaluation dashboard
- Documentation
- Final project quality

---

# Development Principle

Do not build future layers before they are required.

The project should evolve approximately as:

Python
↓
FastAPI
↓
React + FastAPI
↓
LLM
↓
Prompt system
↓
Evals
↓
RAG
↓
Agents
↓
MCP
↓
Production
↓
Observability + full Evals

The frontend and backend repositories follow the same project roadmap,
while each repository owns its respective implementation.
