# AI Studio Backend — Architecture

## Repository and technology

This repository is the FastAPI backend for AI Studio. The React frontend is
maintained separately in `ai-studio-frontend` and consumes these HTTP APIs.

- Python
- FastAPI and Uvicorn
- Pydantic and Pydantic Settings
- Gemini and Groq provider SDKs behind a provider abstraction

## Current architecture

```text
Frontend
↓
FastAPI route and Pydantic request validation
↓
ChatService, ImageService, or AudioService
↓
LLMProvider abstraction
↓
GeminiProvider or GroqProvider
↓
External model API
```

The Prompt Library persistence path is parallel to the existing AI request
flow:

```text
Prompt Library route dependency
↓
get_db_session()
↓
Lazy SQLAlchemy engine and session factory
↓
Neon PostgreSQL
```

Prompt persistence uses a focused layer within that path:

```text
Prompt CRUD route
↓
PromptService (coordination and transaction boundary)
↓
PromptRepository (SQLAlchemy access only)
↓
Session and Neon PostgreSQL
```

Routes remain thin: they validate inputs, resolve services with dependency
injection, and translate expected provider/tool failures into safe HTTP 503
responses. Services coordinate the application workflow. Providers contain SDK
calls, provider-specific message translation, and provider-specific output
handling.

## Current backend structure

```text
app/
├── main.py
├── api/
│   ├── audio.py
│   ├── chat.py
│   ├── evaluations.py
│   ├── images.py
│   └── prompts.py
├── core/
│   ├── config.py
│   └── database.py
├── llm/
│   ├── factory.py
│   ├── gemini.py
│   ├── groq.py
│   └── provider.py
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── prompt.py
│   └── prompt_version.py
├── repositories/
│   ├── __init__.py
│   ├── prompt_repository.py
│   └── prompt_version_repository.py
├── alembic/
│   ├── env.py
│   └── versions/
├── alembic.ini
├── schemas/
│   ├── audio.py
│   ├── chat.py
│   ├── evaluation.py
│   ├── image.py
│   ├── prompt.py
│   └── tool.py
├── services/
│   ├── audio_service.py
│   ├── chat_service.py
│   ├── evaluation_service.py
│   ├── image_service.py
│   └── prompt_service.py
└── tools.py

tests/
```

## API surface

| Endpoint | Request | Response | Purpose |
| --- | --- | --- | --- |
| `GET /health` | — | status object | Health check |
| `POST /chat` | `ChatRequest` | `ChatResponse` | Complete text response |
| `POST /chat/stream` | `ChatRequest` | `text/plain` stream | Plain-text chunks |
| `POST /chat/structured` | `StructuredChatRequest` | `StructuredChatResponse` | Validated `{ answer }` output |
| `POST /chat/formatted` | `FormattedChatRequest` | `FormattedChatResponse` | Requested output format |
| `POST /chat/tools` | `ToolChatRequest` | `ToolChatResponse` | Backend-owned tool demo |
| `POST /images/generate` | `ImageGenerationRequest` | `ImageGenerationResponse` | One generated image |
| `POST /audio/generate` | `AudioGenerationRequest` | `AudioGenerationResponse` | Text-to-speech audio |
| `POST /prompts/build` | `PromptRequest` | `PromptResponse` | Constructed prompt |
| `POST /prompts/experiment` | `PromptExperimentRequest` | `ChatResponse` | Prompt run |
| `POST /prompts/compare` | `PromptComparisonRequest` | `PromptComparisonResponse` | Variant comparison |
| `POST /prompts` | `PromptCreate` | `PromptRead` | Create a persisted prompt |
| `GET /prompts` | Search/filter/pagination query | `PromptPage` | List persisted prompts |
| `GET /prompts/{prompt_id}` | UUID path | `PromptRead` | Read a persisted prompt |
| `PATCH /prompts/{prompt_id}` | `PromptUpdate` | `PromptRead` | Update a persisted prompt |
| `DELETE /prompts/{prompt_id}` | UUID path | 204 | Delete a persisted prompt |
| `GET /prompts/categories` | — | `list[str]` | List distinct prompt categories |
| `PUT /prompts/{prompt_id}/favorite` | — | `PromptRead` | Mark a prompt as favorite |
| `DELETE /prompts/{prompt_id}/favorite` | — | `PromptRead` | Remove a prompt from favorites |
| `PUT /prompts/{prompt_id}/template` | — | `PromptRead` | Mark a prompt as a reusable template |
| `DELETE /prompts/{prompt_id}/template` | — | `PromptRead` | Remove template status |
| `GET /prompts/{prompt_id}/versions` | UUID path | `list[PromptVersionRead]` | List immutable versions, newest first |
| `GET /prompts/{prompt_id}/versions/{version}` | UUID path + integer version | `PromptVersionRead` | Read one immutable version |
| `POST /evaluations/run` | `EvaluationDataset` | `EvaluationRunResponse` | In-memory evaluation run |

Pydantic rejects invalid request bodies with FastAPI validation responses. Text,
image, and audio inputs have a 1–4,000 character limit. Provider failures are
normalized to `LLMProviderError`; routes return a safe 503 rather than exposing
SDK details. The tool route also translates invalid/unknown tool execution into
the same safe 503 response.

## Chat, structured output, and formatted output

All chat flows construct an ordered conversation from completed history plus
the new user message. `ChatRequest` validates the message, history roles and
content, and a finite temperature from 0 to 2.

### Structured output

```text
POST /chat/structured
↓
StructuredChatRequest
↓
ChatService.create_structured_response()
↓
LLMProvider.generate_structured_output()
↓
StructuredChatResponse validation
```

`StructuredChatResponse` is a strict Pydantic model with exactly one required
`answer` string. Both providers request JSON-schema-constrained output and then
run `validate_structured_chat_output()`, which calls
`StructuredChatResponse.model_validate_json()`. Invalid JSON, missing fields,
extra fields, or invalid values become `LLMProviderError` and a safe 503 at the
route boundary.

### Formatted output

`POST /chat/formatted` accepts `output_format` as one of:

| Format | Backend behavior | Response `content` |
| --- | --- | --- |
| `json` | Uses the structured-output flow | `StructuredChatResponse` |
| `markdown` | Adds a Markdown instruction to the user message | string |
| `table` | Adds a Markdown-table instruction | string |
| `email` | Requests a complete email | string |
| `html` | Requests valid HTML only | string |

The response repeats the selected `output_format` and returns either validated
structured JSON for `json` or generated text for the other formats. The
backend owns output selection and validation; rendering is a frontend concern.

## Tool-calling orchestration

`POST /chat/tools` demonstrates one backend-controlled tool call per request.
The frontend sends only the chat request and never selects or executes a tool.

```text
ToolChatRequest
↓
ChatService.create_tool_response()
↓
Provider requests text or one ToolCall
↓
Validate allowlisted tool name and Pydantic arguments
↓
Execute deterministic backend tool
↓
Provider produces final text from the tool result
↓
ToolChatResponse(message, optional tool_name)
```

`ToolModelResponse` Pydantic validation requires exactly one of direct text or
a tool call. The allowlist in `app/tools.py` currently contains `add_numbers`
and `get_project_name`. `add_numbers` validates bounded integer arguments;
both tool argument schemas forbid extra fields. Unknown tools, invalid
arguments, malformed provider calls, and attempts to return multiple calls are
rejected. If the provider returns direct text, no tool runs and `tool_name` is
`null`.

Gemini and Groq translate their native function-calling response into the
provider-neutral tool schemas. After execution, each provider receives its
native tool-result representation and is instructed not to make another tool
call for the final response.

## Image generation

```text
POST /images/generate
↓
ImageGenerationRequest(prompt)
↓
ImageService.generate_image()
↓
LLMProvider.generate_image()
↓
ImageGenerationResponse(image_base64, mime_type)
```

The provider-neutral response returns one Base64-encoded image and its MIME
type, not an SDK object or provider URL. Gemini calls its configured image
model and returns the generated image bytes. Groq explicitly reports image
generation as unsupported, which the route exposes as the standard safe 503.

## Audio / text-to-speech generation

```text
POST /audio/generate
↓
AudioGenerationRequest(text)
↓
AudioService.generate_audio()
↓
LLMProvider.generate_audio()
↓
AudioGenerationResponse(audio_base64, mime_type)
```

The provider-neutral response returns Base64 audio bytes and a MIME type.
Gemini requests an audio modality with the configured model and voice. Groq
uses its speech API and returns WAV audio. The frontend can convert either
response to a native browser audio source without provider-specific handling.

## Provider abstraction

`LLMProvider` defines these operations:

- `generate_text()` and `stream_text()`
- `generate_structured_output()`
- `request_tool_response()` and `generate_text_from_tool_result()`
- `generate_image()`
- `generate_audio()`

The factory selects the active provider using `LLM_PROVIDER` and constructs it
with environment settings. `ChatService`, `ImageService`, and `AudioService`
remain provider-agnostic.

| Capability | Gemini | Groq |
| --- | --- | --- |
| Text generation | Supported | Supported |
| Streaming contract | One generated result yielded as one chunk | Incremental SDK chunks |
| Structured JSON | Provider JSON MIME type and schema | Strict JSON schema response format |
| Tool calling | Native function declarations and responses | Native function tools and tool messages |
| Image generation | Supported with configured image model | Unsupported |
| Audio/TTS | Supported with configured audio model and voice | Supported with configured speech model and voice |

## Configuration

Settings load from environment variables and an optional local `.env` file;
secrets use `SecretStr` and are never committed.

- `FRONTEND_ORIGIN`
- `DATABASE_URL` — existing Neon PostgreSQL connection string
- `LLM_PROVIDER`, `LLM_MODEL`
- `GEMINI_API_KEY`, `GEMINI_IMAGE_MODEL`, `GEMINI_AUDIO_MODEL`,
  `GEMINI_AUDIO_VOICE`
- `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_AUDIO_MODEL`, `GROQ_AUDIO_VOICE`

The selected provider must support the requested capability. For example,
using `LLM_PROVIDER=groq` for image generation returns the documented safe
503 because Groq does not provide image generation in this implementation.

## Database foundation

Sprint 6 uses the existing `DATABASE_URL` for Neon PostgreSQL. The setting is
secret-backed and optional at application configuration time so existing
non-persistence routes can continue to run without opening a database
connection. Database use fails clearly when a session is requested without a
configured URL.

`app/core/database.py` normalizes standard `postgresql://` and legacy
`postgres://` URLs to SQLAlchemy's `postgresql+psycopg://` dialect, creates an
engine with connection-pool pre-ping, and exposes a cached session factory plus
the `get_db_session()` dependency. Engine creation is lazy and does not connect
to Neon at import time.

`app/models/base.py` defines the shared SQLAlchemy `DeclarativeBase`. Alembic
is configured in `alembic/` with
`Base.metadata` as its `target_metadata`, so `alembic revision --autogenerate`
will discover future model tables. Its `env.py` calls `get_database_url()` from
`app.core.database`, which reads the secret-backed `DATABASE_URL` setting and
applies the existing psycopg dialect normalization. `alembic.ini` contains no
credentials.

Alembic uses the same synchronous SQLAlchemy/psycopg architecture as the
application. Online migrations create a short-lived `NullPool` engine; offline
migrations render SQL without opening a database connection. The first
model-backed revision is applied; the Prompt CRUD API consumes that foundation
through the request-scoped session dependency.

For subsequent model changes, export the model through `app.models`, run
`alembic revision --autogenerate -m "describe change"`, then run
`alembic upgrade head`. Use `alembic downgrade -1` (or a specific revision) to
revert a revision. These commands require `DATABASE_URL`; no URL is stored in
`alembic.ini`.

### Prompt persistence

`app.models.prompt.Prompt` is the first persistent prompt-library entity and
is imported from `app.models`, ensuring it is registered on `Base.metadata` for
Alembic. The `prompts` table was created by revision `3756e55c0b52`.

| Field | Storage and behavior |
| --- | --- |
| `id` | PostgreSQL UUID primary key; generated by the application with `uuid4` |
| `title` | Required `VARCHAR(255)`; database constraint rejects blank values |
| `content` | Required `TEXT`; database constraint rejects blank values |
| `description` | Optional `TEXT` |
| `category` | Required `VARCHAR(100)`, defaults to `general`, indexed |
| `is_favorite` / `is_template` | Required booleans, default `false` |
| `created_at` | Required timezone-aware timestamp, defaults to database `now()`, indexed |
| `updated_at` | Required timezone-aware timestamp, defaults to database `now()` and is refreshed by SQLAlchemy on ORM updates |

Prompt persistence is available through `PromptRepository` and `PromptService`;
the Prompt CRUD API uses `get_db_session()` to construct them once per request.

`PromptRepository` owns only SQLAlchemy operations: create, lookup by ID, list
ordered by newest creation time, update, and delete. It flushes mutations so
database-generated values and constraint errors are available within the active
unit of work, but it never commits. `PromptService` accepts `PromptCreate` and
`PromptUpdate` schemas, coordinates repository calls, commits successful
mutations, and rolls back if a commit fails. Read operations do not commit or
mutate persistence state.
`PromptRead` is the ORM-compatible response schema at the API boundary.

`POST /prompts` returns 201, reads and updates return 200, and successful
deletes return 204 with no response body. Missing IDs return a consistent 404;
request validation uses FastAPI's normal 422 response behavior. Semantic/vector
search remains outside the current API scope.

`PATCH /prompts/{prompt_id}` assigns or changes `category`, `is_favorite`, and
`is_template` with the other editable prompt fields. The favorite and template
subresource routes are idempotent convenience operations that set their
respective existing boolean fields to `true` or `false`; they do not create
additional tables. `GET /prompts/categories` is a repository-backed distinct,
alphabetically ordered list of stored category values.

`GET /prompts` accepts optional `search`, `category`, `is_favorite`,
`is_template`, `page` (default 1), and `page_size` (default 20, maximum 100)
query parameters. Blank `search` and `category` values are normalized away.
The response is `PromptPage` with `items`, `page`, `page_size`, `total`, and
`total_pages`. `PromptRepository` builds composable PostgreSQL conditions once,
uses `ILIKE` across title/content/description for search, then applies those
same conditions to both the newest-first result query and its total-count
query. Offset/limit pagination occurs in PostgreSQL.

### Prompt versions

`PromptVersion` is a one-to-many immutable historical-content relationship
from `Prompt`. Each row has its own UUID, the prompt foreign key with `ON
DELETE CASCADE`, a positive version number, historical `content`, and a
creation timestamp. A unique `(prompt_id, version)` constraint prevents
duplicate version numbers for a prompt. Title, description, category, favorite,
and template status remain current `Prompt` metadata and are not versioned.

`PromptService` creates V1 in the same transaction as a new Prompt. It creates
a new version only when an update actually changes `content`: it first applies
every supplied field to the current Prompt, then asks
`PromptVersionRepository` for the next version number and snapshots the new
content before committing once. Metadata-only updates (`title`, `description`,
`category`, `is_favorite`, and `is_template`) do not create versions. A no-op
update creates no snapshot or transaction. Any flush, snapshot, or commit
failure rolls back the full mutation.

The `a9f509f6feed` migration created `prompt_versions` and backfilled every
existing prompt as V1 without deleting or modifying prompt rows. The
`9c4e7b1a2d6f` migration removes obsolete version metadata columns, leaving
historical content only. Backfilled versions reuse their source prompt UUID as
the snapshot ID and preserve the original prompt creation timestamp; new
snapshots use application-generated UUIDs. Version list results are ordered
newest version first.
