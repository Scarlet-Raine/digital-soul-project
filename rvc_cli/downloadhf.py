from huggingface_hub import snapshot_download

repo_id = "IAHispano/Applio"
revision = "main"  # or any specific commit hash
local_dir = "C:/Users/EVO/Documents/AI/rvc-cli/download"  # Optional: where to save the folder

snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir)