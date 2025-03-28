import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define constants at the start
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-3\1'
log_filename = 'smollm2.log'
model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
cache_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-3\1'

# Function to set up logging
def setup_logging():
    try:
        os.makedirs(log_directory, exist_ok=True)
        with open(os.path.join(log_directory, log_filename), 'w') as log_file:
            log_file.write("# Log file for SmolLM2 text generation\n")
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
    logging.debug("Starting model and tokenizer loading for text generation.")

    # Load tokenizer
    try:
        logging.debug("Loading tokenizer for model: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_directory
        )
        logging.debug("Tokenizer loaded successfully.")
    except Exception as e:
        terminate_script(f"Error loading tokenizer: {e}")

    # Load model
    try:
        logging.debug("Loading model: %s", model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_directory
        )
        logging.debug("Model loaded successfully.")
    except Exception as e:
        terminate_script(f"Error loading model: {e}")

    # Text Generation Phase
    prompt = "Hello, how can I assist you today?"
    try:
        logging.debug("Tokenizing prompt: %s", prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        logging.debug("Generating text output from model.")
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.debug("Generated text: %s", generated_text)
    except Exception as e:
        terminate_script(f"Error during text generation: {e}")

    # Print and log generated text
    print("Generated Response:\n", generated_text)
