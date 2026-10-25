from dataclasses import dataclass, field


@dataclass
class DocumentPage:
    text: str
    page_number: int
    source: str


@dataclass
class DocumentChunk:
    id: str
    text: str
    document_id: str
    page_number: int
    source: str
    chunk_number: int
    metadata: dict = field(default_factory=dict)

@dataclass
class RetrievalResult:
    chunk_id: str
    document_id: str
    text: str
    source: str
    page_number: int
    similarity: float


@dataclass
class ContextSource:
    label: str
    chunk_id: str
    document_id: str
    filename: str
    page_number: int


@dataclass
class AssembledContext:
    text: str
    sources: list[ContextSource]
    estimated_tokens: int


@dataclass
class RAGPipelineResult:
    query: str
    retrieved_count: int
    context: AssembledContext
    status: str

@dataclass
class BM25Result:
    chunk_id: str
    document_id: str
    text: str
    source: str
    page_number: int
    score: float

@dataclass
class RankedCandidate:
    chunk_id: str
    document_id: str
    text: str
    source: str
    page_number: int
    score: float

@dataclass
class RetrievalPlan:
    strategy: str
    use_reranker: bool
    use_mmr: bool
    reason: str
@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class LLMGenerationResult:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class ResolvedCitation:
    label: str
    chunk_id: str
    document_id: str
    filename: str
    page_number: int

@dataclass
class RAGAnswer:
    query: str
    answer: str
    status: str

    sources: list[ContextSource] = field(
        default_factory=list
    )

    citations: list[ResolvedCitation] = field(
        default_factory=list
    )

    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class CitationValidationResult:
    valid: bool
    citations: list[ResolvedCitation] = field(
        default_factory=list
    )
    invalid_labels: list[str] = field(
        default_factory=list
    )
    has_citations: bool = False
