import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_logging():
    """
    Sets up logging for model loading.
    Creates a log directory if it doesn't exist and configures logging.
    """
    log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\UDEMY_PROJECT-1\12'
    log_filename = 'model_loading.log'
    os.makedirs(log_directory, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=os.path.join(log_directory, log_filename),
        filemode='w'
    )
    logging.info("Logging setup completed.")

def load_model(model_name, cache_dir):
    """
    Loads the tokenizer and model from the specified model repository.
    
    Args:
        model_name (str): The Hugging Face model repository name.
        cache_dir (str): The directory where model files are cached.
    
    Returns:
        tokenizer, model: The loaded tokenizer and model instances.
    """
    logging.info("Starting model loading process for: %s", model_name)
    
    # Load the tokenizer
    try:
        logging.info("Checking for tokenizer in cache directory: %s", cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        logging.info("Tokenizer loaded successfully from %s", model_name)
    except Exception as e:
        logging.error("Error loading tokenizer: %s", str(e))
        raise

    # Load the model
    try:
        logging.info("Checking for model in cache directory: %s", cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        logging.info("Model loaded successfully from %s", model_name)
    except Exception as e:
        logging.error("Error loading model: %s", str(e))
        raise

    return tokenizer, model

if __name__ == "__main__":
    setup_logging()
    
    model_name = "HuggingFaceTB/SmolLM2-360M"
    cache_dir = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\UDEMY_PROJECT-1\12'
    
    logging.info("Initiating model loading for %s", model_name)
    tokenizer, model = load_model(model_name, cache_dir)
    logging.info("Model and tokenizer loaded successfully. Ready for generation.")
    
    print("Model and tokenizer loaded successfully.")
