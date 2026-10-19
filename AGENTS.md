# AI Studio Backend — Codex Instructions

## Project

This repository contains the backend for AI Studio.

The backend is a production-oriented AI engineering platform built
incrementally alongside the AI learning roadmap.

## Technology

- Python
- FastAPI
- Uvicorn
- Pydantic

## Backend Responsibilities

The backend owns:

- API endpoints
- Request/response validation
- Business logic
- LLM integration
- Prompt processing
- RAG
- Vector database integration
- Agents
- Tools
- MCP
- Evals
- Observability

The frontend must not contain these responsibilities.

## Architecture

Frontend
↓
HTTP API
↓
FastAPI routes
↓
Services
↓
LLM provider abstraction
↓
Gemini or Groq API

Keep API routing separate from business logic.

## Current Structure

app/
├── main.py
├── api/
├── core/
├── llm/
├── models/
├── schemas/
└── services/

tests/

Additional directories should only be introduced when the corresponding
feature requires them.

Do not create future AI architecture prematurely.

## LLM Integration

- `LLMProvider` is the provider contract for text generation, structured
  generation, streaming, tool calling, image generation, and audio generation.
- Keep provider SDK calls and message translation inside `app/llm/`.
- Select Gemini or Groq through the provider factory and environment settings.
- Keep `ChatService` provider-agnostic.
- `POST /chat` returns a complete response; `POST /chat/stream` streams plain
  text. Do not introduce SSE, WebSockets, or frontend streaming behavior unless
  a future feature requires it.

## Sprint 4 Prompt Experiments and Evals

- Keep zero-shot, one-shot, and few-shot validation in prompt schemas; reuse
  the prompt builder and `ChatService` for experiments and comparisons.
- `POST /prompts/compare` evaluates the same input against labeled prompt
  configurations and returns their prompts and outputs.
- `POST /evaluations/run` accepts request-scoped evaluation datasets and runs
  cases through `ChatService`.
- Evaluation criteria are descriptive. Current `passed` results use only
  case-insensitive, trimmed expected-versus-actual string equality.
- Do not add persistence, LLM-as-a-Judge, criteria interpretation, or advanced
  evaluation frameworks unless a future sprint requires them.

## Development Rules

1. Use Python.
2. Use FastAPI for HTTP APIs.
3. Use Pydantic models for API request/response contracts.
4. Keep API routes thin.
5. Put business logic in services.
6. Keep configuration in `core/`.
7. Keep database models in `models/`.
8. Keep API schemas in `schemas/`.
9. Never hardcode secrets.
10. Use environment variables for configuration.
11. Do not introduce unnecessary dependencies.
12. Do not refactor unrelated code.
13. Prefer small, incremental changes.
14. Add tests for important backend behaviour.
15. Maintain clear separation between API, services and infrastructure.

## API Design

API endpoints should have:

- Clear HTTP methods
- Explicit request models
- Explicit response models
- Predictable error responses

The frontend communicates with the backend only through defined API
contracts.

## Sprint 5 — Structured AI

- Keep structured JSON generation validated against `StructuredChatResponse`.
- Keep formatted output selection, tool choice/execution, and provider-specific
  media generation in backend services and providers.
- `POST /chat/tools` is a single-tool, allowlisted demo. The backend validates
  tool-call arguments and produces the final response after tool execution.
- `POST /images/generate` and `POST /audio/generate` return provider-neutral
  Base64 bytes with a MIME type; do not return provider SDK objects.
- Gemini supports structured output, tools, image generation, and audio/TTS
  when its relevant settings are configured. Groq supports structured output,
  tools, and audio/TTS, but not image generation.

## Sprint 6 — Prompt Library

- Use the existing Neon PostgreSQL connection through `DATABASE_URL`.
- Keep database configuration and lazy engine/session setup in `app/core/`.
- Put future SQLAlchemy models under `app/models/` and inherit from `Base`.
- Do not connect to Neon during module import or tests that only validate setup.
- Use Alembic for schema revisions. Its `env.py` must resolve `DATABASE_URL`
  through the existing database helpers and target `Base.metadata`.
- The `Prompt` model is the current prompt-library entity. `PromptRepository`
  owns its SQLAlchemy access; `PromptService` coordinates it and owns mutation
  commits/rollbacks. Prompt CRUD routes use `get_db_session` to construct this
  chain. Favorites, categories, and templates are current fields on `Prompt`,
  not separate resources. Prompt list search, filtering, and pagination belong
  in `PromptRepository`.
- `PromptVersion` stores immutable historical `content` only. Create V1 with a
  new prompt and create one new version only when `content` actually changes.
  Changes to title, description, category, favorite, or template status never
  create a version; favorite and template status are not historical state.
  Keep semantic/vector search separate until required.

## Current Project Phase

Current:

Sprint 6 — Prompt Library: Complete

Completed learning:

- Python
- FastAPI
- AI Module 1
- AI Module 2

Sprint 4 delivered:

- Zero-shot, one-shot, and few-shot prompt experiments
- Prompt/output comparison across configurations
- Evaluation datasets and basic evaluation runs
- Deterministic expected-versus-actual pass/fail results

Sprint 5 delivered:

- Pydantic-validated structured responses and formatted output API
- JSON, Markdown, Table, Email, and HTML output formats
- Provider-agnostic function/tool calling orchestration
- Provider-neutral image generation and audio/TTS APIs
- Gemini and Groq provider implementations for Sprint 5 capabilities

Sprint 6 delivered:

- Neon PostgreSQL configuration, SQLAlchemy engine/session foundation, and
  declarative model base
- Alembic configuration using the shared Neon URL and model metadata
- Core `Prompt` model and the applied `prompts` table migration
- Prompt repository/service persistence layers and create/update/read schemas
- Prompt Library CRUD API backed by request-scoped database sessions
- Prompt favorites, categories, and reusable-template API support
- Database-backed prompt search, filtering, and pagination
- Immutable content-only Prompt version history with Version 1 backfill for
  existing prompts

Next:

Sprint 7 — RAG + RAG Evals

## Codex Behaviour

Before implementing a feature:

1. Read this file.
2. Read the relevant documentation under `docs/`.
3. Inspect the existing implementation.
4. Understand the existing API contract.
5. Make the smallest appropriate change.
6. Do not invent future architecture.
7. Do not modify unrelated functionality.
8. Explain significant architectural changes before implementing them.

The project should evolve incrementally with the learning roadmap.
