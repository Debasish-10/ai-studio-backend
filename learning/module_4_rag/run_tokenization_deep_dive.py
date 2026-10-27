
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer


cross_encoder_tokenizer = AutoTokenizer.from_pretrained(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

# model = SentenceTransformer(
#     "sentence-transformers/all-MiniLM-L6-v2"
# )

# tokenizer = model.tokenizer

# text = (
#     "Employees can request reimbursement "
#     "under policy HR-2026-LV-043."
# )

# encoded = tokenizer(
#     text,
#     add_special_tokens=True,
#     truncation=False
# )

# tokens = tokenizer.convert_ids_to_tokens(
#     encoded["input_ids"]
# )

# for token, token_id in zip(
#     tokens,
#     encoded["input_ids"]
# ):
#     print(f"{token:20} -> {token_id}")


# texts = [
#     "Annual leave is 24 days.",
#     "Employees can work remotely two days per week.",
#     "Sick leave."
# ]

# batch = tokenizer(
#     texts,
#     padding=True,
#     truncation=True,
#     max_length=model.max_seq_length,
#     return_tensors="pt"
# )

# print("Input shape:", batch["input_ids"].shape)

# for i, text in enumerate(texts):
#     print("\nText:", text)
#     print("IDs:", batch["input_ids"][i].tolist())
#     print("Mask:", batch["attention_mask"][i].tolist())



# long_text = (
#     "Employees receive 24 days of annual leave. "
#     * 100
# )

# full_encoding = tokenizer(
#     long_text,
#     truncation=False,
#     add_special_tokens=True
# )

# limited_encoding = tokenizer(
#     long_text,
#     truncation=True,
#     max_length=model.max_seq_length,
#     add_special_tokens=True
# )

# print("Original token count:",
#       len(full_encoding["input_ids"]))

# print("After truncation:",
#       len(limited_encoding["input_ids"]))

# print("Tokens excluded:",
#       len(full_encoding["input_ids"])
#       - len(limited_encoding["input_ids"]))


query = "When does unused annual leave expire?"

passage = (
    "Employees may carry forward unused annual leave. "
    "Carried-forward leave must be used before "
    "September 30 of the following year."
)

encoded_pair = cross_encoder_tokenizer(
    query,
    passage,
    add_special_tokens=True,
    truncation=False
)

tokens = cross_encoder_tokenizer.convert_ids_to_tokens(
    encoded_pair["input_ids"]
)

print("Pair tokens:")
print(tokens)

print("\nTotal tokens:")
print(len(tokens))

print("\nToken type IDs:")
print(encoded_pair.get("token_type_ids"))
