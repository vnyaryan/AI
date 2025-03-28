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

def generate_text(tokenizer, model, prompt, max_length=100, num_beams=4, no_repeat_ngram_size=2, early_stopping=True):
    """
    Generates text based on a given prompt.
    
    Args:
        tokenizer: The tokenizer instance.
        model: The model instance.
        prompt (str): The input prompt.
        max_length (int): The maximum length of the generated text.
        num_beams (int): The number of beams for beam search.
        no_repeat_ngram_size (int): The size of n-grams that cannot be repeated.
        early_stopping (bool): Whether to stop generation early if a stopping criterion is met.
    
    Returns:
        str: The generated text.
    """
    logging.info("Generating text for prompt: %s", prompt)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Generate text
    try:
        outputs = model.generate(**inputs, max_length=max_length, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size, early_stopping=early_stopping)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info("Text generated successfully.")
        return generated_text
    except Exception as e:
        logging.error("Error generating text: %s", str(e))
        raise

if __name__ == "__main__":
    setup_logging()
    
    model_name = "HuggingFaceTB/SmolLM2-360M"
    cache_dir = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\UDEMY_PROJECT-1\12'
    
    logging.info("Initiating model loading for %s", model_name)
    tokenizer, model = load_model(model_name, cache_dir)
    logging.info("Model and tokenizer loaded successfully. Ready for generation.")
    
    # Example usage: Generate text based on a prompt
    prompt = "Hello, how are you?"
    generated_text = generate_text(tokenizer, model, prompt)
    print("Generated Text:", generated_text)
