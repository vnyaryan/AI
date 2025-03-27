import os
from huggingface_hub import snapshot_download

# Define the download path
download_path = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\SLIM-MODEL"

# Create the directory if it doesn't exist
if not os.path.exists(download_path):
    os.makedirs(download_path)

try:
    # Download the model repository
    snapshot_download(
        "llmware/slim-ner-tool",
        local_dir=download_path,
        local_dir_use_symlinks=False,
        resume_download=True,
        force_download=True  # Ensures fresh download
    )
    print(f"Model successfully downloaded to {download_path}")

except Exception as e:
    print(f"An error occurred: {e}")
