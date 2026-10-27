
from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

tokenizer = model.tokenizer


def inspect(text: str):
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False
    )

    count = len(encoding["input_ids"])

    return {
        "characters": len(text),
        "tokens": count,
        "within_limit": count <= model.max_seq_length
    }


samples = {
    "Normal text":
        "Employees receive 24 days of annual leave.",

    "Policy ID":
        "HR-2026-LV-043",

    "Technical identifier":
        "xbert_show_onboarding_checklist",

    "Long paragraph":
        "Employees may request annual leave. " * 100
}

for name, text in samples.items():
    result = inspect(text)

    print(f"\n{name}")
    print("Characters:", result["characters"])
    print("Tokens:", result["tokens"])
    print("Within limit:", result["within_limit"])
