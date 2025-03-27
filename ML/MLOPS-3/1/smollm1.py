import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define constants at the start
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-3\1'
log_filename = 'smollm1.log'
model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
cache_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-3\1'

# Function to set up logging
def setup_logging():
    try:
        os.makedirs(log_directory, exist_ok=True)
        with open(os.path.join(log_directory, log_filename), 'w') as log_file:
            log_file.write("# Log file for SmolLM2 model download\n")
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(log_directory, log_filename),
            filemode='a'
        )
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        raise

# Termination function for graceful exits
def terminate_script(message):
    logging.error(message)
    raise SystemExit(message)

if __name__ == "__main__":
    setup_logging()
    logging.debug("Starting the model download process.")

    # Force download tokenizer with resume enabled
    try:
        logging.debug("Downloading tokenizer for model: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_directory,
            force_download=True,  # Force re-download to avoid cache corruption
            resume_download=True  # Resume download if interrupted
        )
        logging.debug("Tokenizer downloaded successfully.")
    except Exception as e:
        terminate_script(f"Error loading tokenizer: {e}")

    # Force download model with resume enabled
    try:
        logging.debug("Downloading model: %s", model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_directory,
            force_download=True,  # Force re-download
            resume_download=True  # Resume download if interrupted
        )
        logging.debug("Model downloaded successfully.")
    except Exception as e:
        terminate_script(f"Error loading model: {e}")

    print("Model and tokenizer downloaded successfully.")
