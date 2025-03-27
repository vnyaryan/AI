import os
import logging
from llmware.models import ModelCatalog

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
LOG_DIRECTORY = r"C:\Users\ARYAVIN\Documents\REPO\AI\ML\MLOPS-4\SLIM-MODEL\log"
LOG_FILENAME = "slim.log"
LOCAL_MODEL_PATH = r"C:\Users\ARYAVIN\Documents\REPO\AI\ML\MLOPS-4\SLIM-MODEL"
MODEL_NAME = "slim-nli-tool"

# ----------------------------------------------------------------------------
# LOGGING SETUP
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
# PERFORM MODEL INFERENCE (Natural Language Inference)
# ----------------------------------------------------------------------------
def perform_inference(model, evidence, conclusion):
    try:
        logging.info("Starting model inference.")
        combined_text = f"Evidence: {evidence}\nConclusion: {conclusion}"

        response = model.function_call(
            combined_text,
            params=["evidence"],
            function="classify"
        )

        logging.info("Inference completed successfully.")
        logging.debug(f"Inference response: {response}")

        inference_result = response.get('llm_response', {})
        logging.info(f"Inference Result: {inference_result}")

        return inference_result

    except Exception as e:
        logging.error(f"Error during model inference: {e}")
        raise

# ----------------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------------
def main():
    setup_logging()

    evidence_text = "The stock market declined yesterday as investors worried increasingly about the slowing economy."
    conclusion_text = "Investors are positive about the market."

    try:
        model = load_slim_model(MODEL_NAME, LOCAL_MODEL_PATH)
        result = perform_inference(model, evidence_text, conclusion_text)

        # Display inference result on the console
        print("\nInference Result:")
        for outcome in result.get("evidence", []):
            print(f"- Relationship: {outcome}")

        logging.info("Script executed successfully.")

    except Exception as e:
        logging.error(f"Script terminated due to an exception: {e}")
        print(f"Script terminated due to an exception: {e}")

# ----------------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
