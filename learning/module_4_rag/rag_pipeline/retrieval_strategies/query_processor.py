
import re


class QueryProcessor:

    def __init__(self):
        self.synonyms = {
            "wfh": "work from home",
            "pto": "paid time off",
        }

    def normalize(self, query: str) -> str:
        return " ".join(query.strip().split())

    def rewrite(
        self,
        query: str,
        previous_topic: str | None = None
    ) -> str:

        query = self.normalize(query)

        if not query:
            raise ValueError("Query cannot be empty.")

        rewritten = query

        # Replace known abbreviations using word boundaries.
        for term, replacement in self.synonyms.items():
            rewritten = re.sub(
                rf"\b{re.escape(term)}\b",
                replacement,
                rewritten,
                flags=re.IGNORECASE
            )

        # Demonstration: resolve one known follow-up.
        if (
            previous_topic == "annual_leave"
            and re.search(
                r"\bcarry\s+(?:them\s+)?forward\b",
                rewritten,
                flags=re.IGNORECASE
            )
            and not re.search(
                r"\bleave\b",
                rewritten,
                flags=re.IGNORECASE
            )
        ):
            return (
                "Can employees carry forward "
                "unused annual leave days?"
            )

        return rewritten



processor = QueryProcessor()

print(processor.rewrite("What is the WFH policy?"))

print(processor.rewrite(
    "Can we carry them forward?",
    previous_topic="annual_leave"
))