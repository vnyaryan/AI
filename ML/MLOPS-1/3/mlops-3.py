"""
-----------------------------------------------------------------------------------------
Script: Fine-Tuning SmolLM for Binary Classification

Description:
This script fine-tunes a SmolLM-based transformer model for binary classification.
It follows a structured approach involving text preprocessing, tokenization, dataset 
preparation, and model fine-tuning.

Step-by-Step Logic:
1. **Setup Logging**:
   - Initializes logging to store debugging and execution details.
   - Logs are stored in the specified directory.

2. **Text Preprocessing**:
   - Cleans input text (removes special characters, converts to lowercase).
   - Removes stopwords (common English words that do not add meaning).
   - Applies lemmatization (converts words to their base form).

3. **Load Tokenizer and Model**:
   - Loads the SmolLM tokenizer and model for binary classification.
   - Ensures special tokens (e.g., PAD, BOS, EOS) are correctly handled.
   - Adjusts model embeddings to recognize new special tokens.

4. **Tokenization**:
   - Converts preprocessed text into tokenized representations.
   - Applies padding and truncation to ensure uniform sequence length.

5. **Dataset Preparation**:
   - Splits tokenized data into training and validation datasets.
   - Creates a Hugging Face dataset for fine-tuning.

6. **Fine-Tune the Model**:
   - Uses the Trainer class to train the model on the prepared dataset.
   - Saves the fine-tuned model for future use.

7. **Exception Handling**:
   - Logs any errors encountered during execution.
   - Ensures script robustness by handling exceptions.

Input:
- Predefined input text samples (`INPUT_TEXTS` list).

Output:
- Fine-tuned SmolLM model saved in the specified directory.
- Log file recording all script activities.


-----------------------------------------------------------------------------------------
"""

import os
import re
import logging
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          Trainer,
                          TrainingArguments)
from sklearn.model_selection import train_test_split
from datasets import Dataset
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Constants
INPUT_TEXTS = [
    "AI is transforming industries.",
    "SmolLM is efficient and lightweight.",
    "Lightweight models are the future.",
    "SmolLM delivers quick inference!"
]
TOKENIZER_MODEL = "HuggingFaceTB/SmolLM-135M"
MAX_SEQ_LENGTH = 20
stop_words = set(stopwords.words('english'))


# ------------------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------------------
def setup_logging():
    """
    Sets up logging for the script.

    - Creates a 'logs' directory if it does not exist.
    - Configures logging to write messages to a log file.
    - Uses 'DEBUG' level logging for detailed traceability.

    No Input/Output.
    """
    try:
        os.makedirs("./logs", exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            filename=r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-1\3\logs\mlops-3.log',
            filemode='w',
            format="%(asctime)s [%(levelname)s] %(message)s"
        )
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        raise


# ------------------------------------------------------------------------
# Preprocessing Functions
# ------------------------------------------------------------------------
def clean_text(texts):
    """Cleans input text by removing special characters and converting to lowercase."""
    logging.info("Text cleaning task started.")
    cleaned_texts = [re.sub(r"[^a-zA-Z\s]", "", text).lower().strip() for text in texts]
    logging.info("Text cleaning task completed.")
    return cleaned_texts


def remove_stopwords(texts):
    """Removes stopwords from the text."""
    logging.info("Stopword removal task started.")
    filtered_texts = [' '.join([word for word in text.split() if word not in stop_words]) for text in texts]
    logging.info("Stopword removal task completed.")
    return filtered_texts


def lemmatize_text(texts):
    """Applies lemmatization to convert words to their base form."""
    logging.info("Lemmatization task started.")
    lemmatizer = WordNetLemmatizer()
    lemmatized_texts = [' '.join([lemmatizer.lemmatize(word) for word in text.split()]) for text in texts]
    logging.info("Lemmatization task completed.")
    return lemmatized_texts


# ------------------------------------------------------------------------
# Dataset Preparation
# ------------------------------------------------------------------------
def prepare_dataset(input_ids, attention_masks, labels=None):
    """Prepares a Hugging Face Dataset for training."""
    logging.info("Dataset preparation task started.")
    data_dict = {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_masks.tolist()
    }
    if labels is not None:
        data_dict["labels"] = labels
    logging.info("Dataset preparation task completed.")
    return Dataset.from_dict(data_dict)


# ------------------------------------------------------------------------
# Fine-Tuning Function
# ------------------------------------------------------------------------
def fine_tune_model(train_dataset, val_dataset, model, tokenizer):
    """Fine-tunes a Llama-based model for binary classification."""
    logging.info("Fine-tuning task started.")
    try:
        model.resize_token_embeddings(len(tokenizer))

        training_args = TrainingArguments(
            output_dir=r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-1\3\logs\smollm_results',
            per_device_train_batch_size=8,
            num_train_epochs=3,
            logging_dir=r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-1\3\logs',
            logging_steps=10,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()
        model.save_pretrained(r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-1\3\logs\smollm_fine_tuned_model')
        logging.info("Fine-tuning task completed successfully.")
    except Exception as e:
        logging.error(f"Error in fine-tuning: {e}")
        raise


# ------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------
def main():
    """
    The main function orchestrates the full pipeline:
    - Sets up logging.
    - Performs text preprocessing.
    - Loads tokenizer and model.
    - Tokenizes and prepares datasets.
    - Fine-tunes the model.
    """
    setup_logging()
    logging.info("Script execution started.")

    try:
        # Step 1: Text Preprocessing
        cleaned_texts = lemmatize_text(remove_stopwords(clean_text(INPUT_TEXTS)))

        # Step 2: Load Tokenizer and Model
        logging.info("Loading tokenizer and model.")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, model_max_length=MAX_SEQ_LENGTH)
        model = AutoModelForSequenceClassification.from_pretrained(TOKENIZER_MODEL, num_labels=2)

        # Step 3: Fix Padding Token Issue
        special_tokens = {'pad_token': '[PAD]', 'bos_token': '<s>', 'eos_token': '</s>'}
        tokenizer.add_special_tokens(special_tokens)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Assign eos_token as pad_token if missing

        # Ensure model recognizes pad token
        model.config.pad_token_id = tokenizer.pad_token_id

        # Resize model embeddings to include new special tokens
        model.resize_token_embeddings(len(tokenizer))

        logging.info(f"Tokenizer and model special tokens set. Pad token ID: {model.config.pad_token_id}")

        # Step 4: Tokenization
        logging.info("Tokenization task started.")
        tokenized_data = tokenizer(
            cleaned_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="np",
            add_special_tokens=True
        )
        logging.info("Tokenization task completed.")

        # Step 5: Dataset Preparation
        train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
            tokenized_data["input_ids"], tokenized_data["attention_mask"], [0, 1, 0, 1], test_size=0.2, random_state=42
        )

        train_dataset = prepare_dataset(train_inputs, train_masks, train_labels)
        val_dataset = prepare_dataset(val_inputs, val_masks, val_labels)

        # Step 6: Fine-Tune Model
        fine_tune_model(train_dataset, val_dataset, model, tokenizer)
        logging.info("Script execution completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise



if __name__ == "__main__":
    main()
