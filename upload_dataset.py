from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_file(
    path_or_fileobj="./mitre_technique_descriptions.json",
    path_in_repo="mitre_technique_descriptions.json",
    repo_id="gbellott/test_dataset",
    repo_type="dataset"
)
