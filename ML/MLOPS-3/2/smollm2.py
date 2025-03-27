import os
import logging
from llama_cpp import Llama

# Define constants at the start
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-3\2'
log_filename = 'smollm2.log'
model_file = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-3\2\SmolLM2-1.7B-Instruct-Q4_K_M.gguf"

# Function to set up logging
def setup_logging():
    try:
        os.makedirs(log_directory, exist_ok=True)
        with open(os.path.join(log_directory, log_filename), 'w') as log_file:
            log_file.write("# Log file for llama.cpp GGUF model text generation\n")
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
    logging.debug("Starting LLaMA model loading using llama.cpp for text generation.")

    # Load the model using llama.cpp
    try:
        logging.debug("Loading LLaMA GGUF model from: %s", model_file)
        model = Llama(
            model_path=model_file,
            n_ctx=2048,  # Context window size
            n_threads=4,  # Adjust based on your CPU cores
        )
        logging.debug("LLaMA model loaded successfully using llama.cpp.")
    except Exception as e:
        terminate_script(f"Error loading LLaMA model with llama.cpp: {e}")

    # Text Generation Phase
    prompt = "Hello, how can I assist you today?"
    try:
        logging.debug("Generating text output from LLaMA model.")
        response = model(prompt, max_tokens=100)  # Generate up to 100 tokens
        generated_text = response["choices"][0]["text"]
        logging.debug("Generated text: %s", generated_text)

    except Exception as e:
        terminate_script(f"Error during text generation: {e}")

    # Print and log generated text
    print("Generated Response:\n", generated_text)
