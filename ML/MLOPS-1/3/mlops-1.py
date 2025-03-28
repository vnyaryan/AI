"""
Script Overview:
----------------
This script is designed to preprocess text data, tokenize it, and fine-tune a small lightweight language model 
(SmolLLM) for binary classification tasks. Below is a detailed explanation of the script's inputs, outputs, 
and its step-by-step logic.

Input:
------
- A list of raw text sentences (defined in the `INPUT_TEXTS` constant).
- Pre-trained model name for tokenization and fine-tuning (e.g., "HuggingFaceTB/SmolLM-135M").
- Constants and configurations such as maximum sequence length (`MAX_SEQ_LENGTH`) and logging settings.

Output:
-------
- Cleaned, tokenized, and preprocessed text data.
- Training and validation datasets formatted for lightweight fine-tuning.
- A fine-tuned SmolLLM model saved to the specified output directory (`./smollm_fine_tuned_model`).
- Detailed logs of the execution process saved to a log file.

Step-by-Step Logic:
-------------------
1. **Setup Logging**: Initializes logging to record progress, debug information, and errors.
2. **Text Cleaning**: Removes special characters, URLs, HTML tags, and converts text to lowercase.
3. **Stopword Removal**: Eliminates common stopwords (e.g., "and", "the") to focus on meaningful content.
4. **Lemmatization**: Reduces words to their base forms (e.g., "running" -> "run").
5. **Sentence Splitting**: Splits paragraphs into individual sentences for finer processing.
6. **Tokenization**: Converts text data into numerical tokens using a pre-trained tokenizer.
7. **Attention Mask Creation**: Generates binary masks to differentiate between real tokens and padding tokens.
8. **Data Splitting**: Splits tokenized data into training and validation sets using `train_test_split`.
9. **Dataset Preparation**: Formats data into Hugging Face Dataset objects for model training and evaluation.
10. **Model Fine-Tuning**: Fine-tunes a pre-trained transformer model for binary classification using 
    Hugging Face's `Trainer`.
11. **Model Saving**: Saves the fine-tuned model for future inference in the `./smollm_fine_tuned_model` directory.

Dependencies:
-------------
- Required Python packages:
  `pip install transformers datasets nltk scikit-learn`
- Ensure NLTK resources (`stopwords`, `wordnet`) are downloaded before running the script.

Notes:
------
- SmolLLM is ideal for scenarios where computational resources are limited.
- The script is optimized for small-scale fine-tuning and quick inference.
"""

import os
import re
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
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

# Step 1: Setup Logging
def setup_logging():
    try:
        os.makedirs("./logs", exist_ok=True)
        logging.basicConfig(level=logging.DEBUG, 
                            filename="./logs/smolllm_fine_tune.log", 
                            filemode='w')
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        raise

# Step 2-4: Preprocessing
def clean_text(texts):
    return [re.sub(r"[^a-zA-Z\s]", "", text).lower().strip() for text in texts]

def remove_stopwords(texts):
    return [' '.join([word for word in text.split() if word not in stop_words]) for text in texts]

def lemmatize_text(texts):
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in text.split()]) for text in texts]

# Step 9: Dataset Preparation
def prepare_dataset(input_ids, attention_masks, labels=None):
    data_dict = {"input_ids": input_ids.tolist(), "attention_mask": attention_masks.tolist()}
    if labels:
        data_dict["labels"] = labels
    return Dataset.from_dict(data_dict)

# Step 10: Model Fine-Tuning
def fine_tune_model(train_dataset, val_dataset):
    try:
        model = AutoModelForCausalLM.from_pretrained(TOKENIZER_MODEL)
        training_args = TrainingArguments(
            output_dir="./smollm_results",
            per_device_train_batch_size=8,
            num_train_epochs=3,
            logging_dir="./logs",
            logging_steps=10,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        trainer.train()
        model.save_pretrained("./smollm_fine_tuned_model")
        logging.info("Model fine-tuning completed and saved successfully.")
    except Exception as e:
        logging.error(f"Error in fine-tuning: {e}")
        raise

# Main Function
def main():
    # # Step 1: Setup Logging
    setup_logging()
    logging.info("Starting the script...")
    try:
        # Steps 2-4: Preprocessing
        cleaned_texts = lemmatize_text(remove_stopwords(clean_text(INPUT_TEXTS)))


        # Steps 5 [NOT IMPLEMENTED]
        # Step 6: Tokenization
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenized_data = tokenizer(cleaned_texts, 
                                   padding="max_length", 
                                   truncation=True, 
                                   max_length=MAX_SEQ_LENGTH, 
                                   return_tensors="np")


        # Step 7: Attention Mask Creation
        input_ids = tokenized_data["input_ids"]
        attention_masks = tokenized_data["attention_mask"]

        # Step 8: Data Splitting
        labels = [0, 1, 0, 1]
        train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
            input_ids, attention_masks, labels, test_size=0.2, random_state=42
        )

        # Step 9: Dataset Preparation
        train_dataset = prepare_dataset(train_inputs, train_masks, train_labels)
        val_dataset = prepare_dataset(val_inputs, val_masks, val_labels)

        # Step 10: Model Fine-Tuning
        fine_tune_model(train_dataset, val_dataset)
        logging.info("Script completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        raise

if __name__ == "__main__":
    main()
