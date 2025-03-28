from huggingface_hub import snapshot_download
import os

download_path = "/tmp/zen" # Replace with your desired path

# Create the directory if it doesn't exist
if not os.path.exists(download_path):
    os.makedirs(download_path)

try:
    snapshot_download(
        "llmware/bling-phi-3-gguf",
        local_dir=download_path,
        local_dir_use_symlinks=False,
        resume_download=True,
        force_download=True  # Add this line
    )
    print(f"Model successfully downloaded to {download_path}")

except Exception as e:
    print(f"An error occurred: {e}")
