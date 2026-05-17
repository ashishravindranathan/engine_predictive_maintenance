import json
import os
from pathlib import Path

from huggingface_hub import HfApi

SPACE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SPACE_DIR / "deployment_config.json"
CONFIG = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

HF_SPACE_REPO_ID = os.getenv("HF_SPACE_REPO_ID") or CONFIG["HF_SPACE_REPO_ID"]
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_SPACE_REPO_ID:
    raise ValueError("HF_SPACE_REPO_ID is not set and not found in deployment_config.json.")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required.")

api = HfApi(token=HF_TOKEN)

api.create_repo(
    repo_id=HF_SPACE_REPO_ID,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True
)

api.upload_folder(
    folder_path=str(SPACE_DIR),
    repo_id=HF_SPACE_REPO_ID,
    repo_type="space",
    ignore_patterns=[
        "__pycache__",
        "*.pyc",
        ".git",
        ".ipynb_checkpoints",
        "logs"
    ]
)

print(f"Deployment files pushed successfully to: https://huggingface.co/spaces/{HF_SPACE_REPO_ID}")
