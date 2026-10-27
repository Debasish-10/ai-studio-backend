
from huggingface_hub import HfApi, snapshot_download
from sentence_transformers import SentenceTransformer


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

api = HfApi()
info = api.model_info(MODEL_ID)

# Demonstration: pin the revision observed now.
# In production, store an approved commit SHA
# explicitly in version-controlled configuration.
approved_revision = info.sha

model_path = snapshot_download(
    repo_id=MODEL_ID,
    revision=approved_revision
)

model = SentenceTransformer(model_path)

print("Model:", MODEL_ID)
print("Revision:", approved_revision)
print("Snapshot:", model_path)
print("Dimensions:", model.get_sentence_embedding_dimension())
