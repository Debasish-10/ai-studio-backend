from app.schemas.chat import ChatResponse
from app.schemas.evaluation import EvaluationDataset
from app.services.evaluation_service import run_evaluation


class StubChatService:
    def __init__(self) -> None:
        self.received_messages: list[str] = []
        self.received_temperatures: list[float] = []

    def create_response(
        self,
        message: str,
        history: list[object] | None = None,
        temperature: float = 1.0,
    ) -> ChatResponse:
        self.received_messages.append(message)
        self.received_temperatures.append(temperature)
        return ChatResponse(message=f"Actual output for: {message}")


def test_run_evaluation_returns_expected_and_actual_outputs_in_case_order() -> None:
    chat_service = StubChatService()
    dataset = EvaluationDataset(
        name="Greeting quality",
        cases=[
            {
                "input": "Greet Ada.",
                "expected_output": "Actual output for: Greet Ada.",
                "criteria": ["Uses the name Ada"],
            },
            {
                "input": "Greet Lin.",
                "expected_output": "Hello, Lin!",
                "criteria": ["Uses the name Lin"],
            },
        ],
    )

    response = run_evaluation(dataset, chat_service, temperature=0.35)

    assert response.dataset_name == "Greeting quality"
    assert [result.actual_output for result in response.results] == [
        "Actual output for: Greet Ada.",
        "Actual output for: Greet Lin.",
    ]
    assert [result.expected_output for result in response.results] == [
        "Actual output for: Greet Ada.",
        "Hello, Lin!",
    ]
    assert response.results[0].criteria == ["Uses the name Ada"]
    assert [result.passed for result in response.results] == [True, False]
    assert chat_service.received_messages == ["Greet Ada.", "Greet Lin."]
    assert chat_service.received_temperatures == [0.35, 0.35]
