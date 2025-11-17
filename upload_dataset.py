from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_file(
    path_or_fileobj="./data.parquet",
    path_in_repo="dataset/data.parquet",
    repo_id="gbellott/cse132_final_project",
    repo_type="dataset"
)
