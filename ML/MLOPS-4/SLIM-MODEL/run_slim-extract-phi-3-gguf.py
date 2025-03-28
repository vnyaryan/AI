import os
import logging
from llmware.models import ModelCatalog

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
LOG_DIRECTORY = r"C:\Users\ARYAVIN\Documents\REPO\AI\ML\MLOPS-4\SLIM-MODEL\log"
LOG_FILENAME = "slim.log"
LOCAL_MODEL_PATH = r"C:\Users\ARYAVIN\Documents\REPO\AI\ML\MLOPS-4\SLIM-MODEL"
MODEL_NAME = "slim-extract-phi-3-gguf"

# ----------------------------------------------------------------------------
# LOGGING SETUP (Forced logging configuration)
# ----------------------------------------------------------------------------
def setup_logging():
    try:
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(LOG_DIRECTORY, LOG_FILENAME),
            filemode='w'
        )
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        raise

# ----------------------------------------------------------------------------
# LOAD MODEL FROM LOCAL DIRECTORY
# ----------------------------------------------------------------------------
def load_slim_model(model_name, model_path):
    try:
        logging.info(f"Loading model '{model_name}' from local path '{model_path}'")
        model = ModelCatalog().load_model(model_name, model_folder=model_path)
        logging.info(f"Model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model '{model_name}': {e}")
        raise

# ----------------------------------------------------------------------------
# PERFORM MODEL INFERENCE
# ----------------------------------------------------------------------------
def perform_inference(model, text_sample):
    try:
        logging.info("Starting model inference.")
        response = model.function_call(text_sample)
        logging.info("Inference completed successfully.")
        logging.debug(f"Inference response: {response}")

        summarized_output = response.get('llm_response', [])
        logging.info(f"Summarized Response: {summarized_output}")

        return summarized_output

    except Exception as e:
        logging.error(f"Error during model inference: {e}")
        raise

# ----------------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------------
def main():
    setup_logging()

    text_sample = """
     Major providers like Azure and AWS dominate the cloud computing industry
    """

    try:
        model = load_slim_model(MODEL_NAME, LOCAL_MODEL_PATH)
        summarized_output = perform_inference(model, text_sample)

        # Display summarized output on console
        print("\nSummarized Output:")
        for sentence in summarized_output:
            print(f"- {sentence}")

        logging.info("Script executed successfully.")

    except Exception as e:
        logging.error(f"Script terminated due to an exception: {e}")
        print(f"Script terminated due to an exception: {e}")

# ----------------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
