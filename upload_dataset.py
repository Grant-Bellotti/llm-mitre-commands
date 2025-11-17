from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_file(
    path_or_fileobj="./data.parquet",
    path_in_repo="dataset/data.jsonl",
    repo_id="gbellott/test_dataset",
    repo_type="dataset"
)
