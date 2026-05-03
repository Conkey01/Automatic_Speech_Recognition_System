# upload_to_hf.py
from huggingface_hub import HfApi, upload_file

REPO_ID = "Conkey01/mini-wav2vec2-asr"   # change this!

upload_file(
    path_or_fileobj="./asr_model_deploy.pth",
    path_in_repo="asr_model.pth",
    repo_id=REPO_ID,
    repo_type="model",
)
print(f"✓ Uploaded to https://huggingface.co/{REPO_ID}")
