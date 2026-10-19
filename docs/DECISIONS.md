# AI Studio — Architecture Decisions

## ADR-001 — Separate Frontend and Backend Repositories

### Decision

Use two repositories:

- `ai-studio-frontend`
- `ai-studio-backend`

### Reason

This provides:

- Clear separation of responsibilities
- Independent deployment
- Independent development
- A realistic production-style architecture
- A cleaner portfolio project

---

## ADR-002 — Frontend Technology

### Decision

Use:

- React
- Vite
- TypeScript

### Reason

This is the selected frontend stack for the project.

---

## ADR-003 — Backend Technology

### Decision

Use:

- Python
- FastAPI
- Uvicorn
- Pydantic

### Reason

Python and FastAPI have already been covered as part of the backend
foundation and are appropriate for the AI engineering project.

---

## ADR-004 — Database

### Decision

Use Neon PostgreSQL.

### Reason

PostgreSQL provides the primary relational database for the application while
Neon provides the existing managed PostgreSQL connection used by this project.

---

## ADR-005 — Hosting

### Decision

Frontend:

Vercel

Backend:

Render

Database:

Neon PostgreSQL

### Reason

This provides a simple deployment architecture suitable for development,
learning and portfolio use.

---

## ADR-006 — AI Models

### Decision

Use external model APIs rather than running large models locally.

### Reason

This keeps the project focused on AI engineering concepts such as:

- Prompting
- LLM integration
- RAG
- Agents
- Evals
- Production architecture

Provider selection is configured at runtime; the implemented provider layer is
described in ADR-009.

---

## ADR-007 — Progressive Architecture

### Decision

Do not create the complete final AI architecture at the beginning.

Introduce new layers only when they become necessary.

Examples:

- LLM layer → when LLM integration begins
- RAG → when RAG is introduced
- Agents → when agent development begins
- MCP → when MCP is introduced
- Evals → progressively as evaluation requirements increase
- Observability → during production-oriented development

### Reason

The project should reinforce the learning roadmap rather than hide
concepts behind premature abstractions.

---

## ADR-008 — Environment Variables

### Decision

Use environment variables for configuration and secrets.

Actual environment files are local and must not be committed.

Tracked example files may contain variable names and safe development
defaults, but never real credentials or API keys.

---

## ADR-009 — Provider-Agnostic LLM Integration

### Decision

Use an `LLMProvider` abstraction with a factory that selects Gemini or Groq
from environment configuration.

### Reason

This keeps `ChatService` independent of provider SDKs while keeping provider
configuration and message translation inside provider-specific implementations.

---

## ADR-010 — Plain HTTP Text Streaming

### Decision

Expose streamed model output through `POST /chat/stream` using
`StreamingResponse` with `text/plain` content.

### Reason

This adds incremental output through the existing route and service layers
without introducing SSE, WebSockets, or frontend streaming behavior.

---

## ADR-011 — Prompt Experiment Configurations

### Decision

Represent zero-shot, one-shot, and few-shot experiments as validated prompt
configurations. Reuse the prompt builder and `ChatService` for individual runs
and same-input output comparisons.

### Reason

This keeps prompt construction and model invocation consistent while allowing
configurations to vary in instructions, context, examples, and temperature.

---

## ADR-012 — Deterministic Basic Evaluation Checks

### Decision

Accept request-scoped evaluation datasets and run each case through
`ChatService`. Return expected output, actual output, criteria, and a `passed`
value based on case-insensitive, trimmed string equality.

### Reason

This provides a transparent Sprint 4 baseline without persistence, LLM-as-a-
Judge, criteria interpretation, or advanced evaluation frameworks.

---

## ADR-013 — Pydantic-Validated Structured and Formatted Output

### Decision

Use strict Pydantic response schemas for structured chat output. Expose JSON,
Markdown, Table, Email, and HTML through `POST /chat/formatted`, with JSON
using the structured-output path and text formats using explicit backend-owned
prompt instructions.

### Reason

- Validates the application contract after provider generation
- Keeps output-format selection and prompt construction in the backend
- Gives clients a stable discriminated response shape without embedding
  provider-specific schema behavior in the frontend

---

## ADR-014 — Single Backend-Controlled Tool-Calling Loop

### Decision

Use a provider-neutral `ToolCall` and `ToolModelResponse` contract for a
single, allowlisted tool call per `POST /chat/tools` request. The backend
validates and executes the call, then asks the provider for final text.

### Reason

- Keeps tool selection, argument validation, and execution on the trusted
  backend boundary
- Prevents clients from invoking arbitrary tools or executing provider output
- Establishes a small deterministic orchestration baseline before later agent
  and MCP work

---

## ADR-015 — Provider-Neutral Base64 Media Responses

### Decision

Return generated image and audio bytes as Base64 with an explicit MIME type
through `POST /images/generate` and `POST /audio/generate`.

### Reason

- Avoids exposing provider SDK objects, temporary URLs, or provider-specific
  response shapes to clients
- Lets clients render native image and audio elements consistently
- Preserves capability differences: Gemini supports image and audio generation,
  while Groq supports audio but reports image generation as unsupported

---

## ADR-016 — Lazy SQLAlchemy Foundation for Neon PostgreSQL

### Decision

Use the existing `DATABASE_URL` setting for a lazy synchronous SQLAlchemy
engine and request-scoped sessions. Define future persistence models from a
shared declarative `Base`.

### Reason

- Keeps Neon credentials in the existing settings layer
- Avoids database connections while importing the application or running
  non-persistence tests
- Gives later persistence features one consistent model and session foundation

---

## ADR-017 — Alembic Revisions Use Shared Database Configuration and Metadata

### Decision

Use Alembic with its standard project-local structure. Configure `env.py` to
obtain the normalized Neon URL through `app.core.database.get_database_url()`
and set `target_metadata` to `Base.metadata`. Do not create an empty baseline
revision.

### Reason

- Avoids credentials and connection-string behavior diverging between
  application and migration commands
- Enables reliable autogeneration when the first SQLAlchemy model is added
- Keeps synchronous psycopg migrations consistent with the application
- Prevents a no-op migration from representing application schema before any
  schema exists

---

## ADR-018 — Prompt Library Core Persistence

### Decision

Persist the prompt library's current entry in a single `prompts` table with UUID
identity, title/content, optional description, category, favorite/template
flags, and timestamps. Create the table only through Alembic revisions.

### Reason

- Establishes a useful, constrained persistence model for the Prompt Library
- Avoids creating repositories, services, or CRUD endpoints before their
  respective Sprint 6 work
- Lets Alembic remain the audited source of schema changes

---

## ADR-019 — Prompt Repository and Service Own Separate Persistence Concerns

### Decision

Use `PromptRepository` for SQLAlchemy access and `PromptService` for prompt
operation coordination and transaction completion. Repositories flush but never
commit; the service commits successful create, update, and delete operations
and rolls back when a commit fails. Use Pydantic create, update, and read
schemas at the service/API boundary.

### Reason

- Keeps raw SQLAlchemy queries outside routes and service business rules
- Preserves one clear transaction boundary for a future request-scoped session
- Keeps ORM entities separate from API contracts
- Leaves room for later validation, versioning, and other prompt-library rules
  without changing route responsibilities

---

## ADR-020 — Immutable Prompt Snapshots for Version History

### Decision

Store prompt history in a `prompt_versions` table with V1 at Prompt creation
and one immutable content snapshot for each actual `content` change. Keep the
`prompts` row as the current state; title, description, category, favorite, and
template status are current metadata only and never create or belong in version
records. Use `PromptVersionRepository` for version queries and writes, with
`PromptService` committing the current-row and snapshot mutations together.

### Reason

- Preserves historical prompt content without duplicating mutable library metadata
- Keeps the current Prompt query path simple for CRUD and search
- Makes V1 creation, V2/V3 updates, and migration backfills explicit and
  auditable
- Defers restore, comparison, and version-evaluation workflows until their
  requirements are defined

---
