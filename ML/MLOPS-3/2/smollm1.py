import os
import logging
import subprocess

# Define constants at the start
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-3\2'
log_filename = 'smollm1.log'
model_name = "bartowski/SmolLM2-1.7B-Instruct-GGUF"
cache_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-3\2'
gguf_model_file = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf"

# Function to set up logging
def setup_logging():
    try:
        os.makedirs(log_directory, exist_ok=True)
        with open(os.path.join(log_directory, log_filename), 'w') as log_file:
            log_file.write("# Log file for SmolLM2 GGUF model download\n")
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
    logging.debug("Starting the GGUF model download process.")



    # Download GGUF model using CLI
    try:
        logging.debug("Downloading GGUF model using CLI: %s", model_name)
        command = f"huggingface-cli download {model_name} --include \"{gguf_model_file}\" --local-dir {cache_directory}"
        print(f"Executing command: {command}")
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"Command failed with output: {result.stderr}")
            terminate_script(f"Error downloading GGUF model using CLI: {result.stderr}")
        
        # Check if the file was downloaded
        downloaded_file_path = os.path.join(cache_directory, gguf_model_file)
        if os.path.exists(downloaded_file_path):
            logging.debug("GGUF model downloaded successfully using CLI.")
        else:
            logging.error(f"GGUF model file '{gguf_model_file}' not found in the cache directory after CLI download.")
            # Attempt to download again or use an alternative method
            logging.debug("Attempting alternative download method.")
            terminate_script(f"Error: GGUF model file '{gguf_model_file}' not downloaded using CLI or alternative method.")
    except Exception as e:
        logging.error(f"Error downloading GGUF model using CLI: {e}")
        # Handle the error or attempt an alternative download method
        logging.debug("Attempting alternative download method.")
        terminate_script(f"Error: GGUF model file '{gguf_model_file}' not downloaded using CLI or alternative method.")

    print("GGUF model downloaded successfully.")

