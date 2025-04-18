from huggingface_hub import snapshot_download
import os

model_id = "google/gemma-2-27b-it"
# Define target directory (current directory in this case)
local_dir = "."
# Set cache directory to scratch to avoid filling home during download meta-steps
cache_dir = "/scratch/project_2013894/.cache/huggingface_dl_cache" # Temporary cache
os.makedirs(cache_dir, exist_ok=True)


print(f"Downloading {model_id} to {os.path.abspath(local_dir)}...")

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False, # Download actual files, not symlinks
    resume_download=True,       # Resume if interrupted
    cache_dir=cache_dir,        # Use scratch for temp download cache
    # Optional: Specify token if env var/login isn't working, but login should suffice
    # token="hf_YOUR_TOKEN_HERE"
)

print("Download complete.")
print(f"Model files saved in: {os.path.abspath(local_dir)}")