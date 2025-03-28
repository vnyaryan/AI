"""
Script Overview:
----------------
This script is designed to preprocess text data, tokenize it, and fine-tune a pre-trained transformer model 
for binary classification tasks. Below is a detailed explanation of the script's inputs, outputs, 
and its step-by-step logic.

Input:
------
- A list of raw text sentences (defined in the `INPUT_TEXTS` constant).
- Pre-trained model name for tokenization and fine-tuning (e.g., "bert-base-uncased").
- Constants and configurations such as maximum sequence length (`MAX_SEQ_LENGTH`) and logging settings.

Output:
-------
- Cleaned, tokenized, and preprocessed text data.
- Hugging Face Dataset objects for training and validation.
- A fine-tuned transformer model saved to the specified output directory (`./fine_tuned_model`).
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
11. **Model Saving**: Saves the fine-tuned model for future inference in the `./fine_tuned_model` directory.

Execution:
----------
- The script is executed directly. It processes the input data, fine-tunes the model, and saves the results.
- Logs provide detailed information about each step for debugging and analysis.

Dependencies:
-------------
- Required Python packages:
  `pip install transformers scikit-learn datasets nltk`
- Ensure NLTK resources (`stopwords`, `wordnet`, `punkt`) are downloaded before running the script.

Notes:
------
- Input data and configurations are customizable by modifying the `INPUT_TEXTS` and constants.
- Logs and intermediate results are saved for troubleshooting and reproducibility.
"""


import os  
# Provides functions for interacting with the operating system.
# Why required:
# - Used for file and directory operations, such as creating the log directory if it doesn't exist.

import re  
# Provides support for regular expressions.
# Why required:
# - Used for cleaning text data by removing unwanted patterns like URLs, HTML tags, and non-alphanumeric characters.

import logging  
# Offers a flexible framework for generating log messages.
# Why required:
# - Used to record debug and info-level logs to track the script's progress and trace errors.

# **Third-Party Libraries**

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments  
# Provides tools for working with pre-trained transformer models and fine-tuning them.
# Why required:
# - `AutoTokenizer`: Used for tokenizing text into numerical tokens for model input.
# - `AutoModelForSequenceClassification`: Used to load a pre-trained model for fine-tuning on classification tasks.
# - `Trainer` and `TrainingArguments`: Used to configure and run the model training process.

from sklearn.model_selection import train_test_split  
# A library for machine learning utilities, including splitting data into training and test sets.
# Why required:
# - Used to split the tokenized data into training and validation sets for model training.

from datasets import Dataset  
# Part of Hugging Face's library for working with datasets.
# Why required:
# - Used to create Dataset objects for model training and evaluation.

import nltk  
# A library for natural language processing (NLP).
# Why required:
# - Used for preprocessing text data, such as stopword removal, lemmatization, and sentence tokenization.

from nltk.corpus import stopwords  
# Provides a list of common stopwords for various languages.
# Why required:
# - Used to remove common stopwords from text data, focusing on meaningful content.

from nltk.stem import WordNetLemmatizer  
# Offers tools for lemmatizing words to their base forms.
# Why required:
# - Used to reduce words to their root forms (e.g., "running" -> "run") for better text normalization.

from nltk.tokenize import sent_tokenize  
# Provides methods for splitting text into sentences.
# Why required:
# - Used to split paragraphs into individual sentences for finer text processing.

# Download required NLTK resources
nltk.download('stopwords')
# Downloads the stopwords dataset for stopword removal.

nltk.download('wordnet')
# Downloads the WordNet dataset required for lemmatization.

nltk.download('punkt')
# Downloads the Punkt tokenizer models needed for sentence splitting.

# Constants
INPUT_TEXTS = [
    "I love AI.",
    "Transformers are amazing.",
    "Machine learning is the future.",
    "Natural language processing is fascinating!"
]
# Predefined list of input sentences to be preprocessed and used for fine-tuning the model.

TOKENIZER_MODEL = "bert-base-uncased"
# The pre-trained tokenizer model to be used for tokenizing the text.

MAX_SEQ_LENGTH = 10
# Maximum sequence length for tokenized inputs.

stop_words = set(stopwords.words('english'))
# Set of English stopwords, used to filter out common, less meaningful words.


    Creates the log directory if it does not exist and configures the logging settings 
    to save logs in the specified file with detailed debug information.

    Raises:
        Exception: If there's an error while setting up logging.
    """
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-1\2'
        log_filename = 'mlops-5.log'

        os.makedirs(log_directory, exist_ok=True)

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='w')
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        raise

# Text cleaning functions
def clean_text(texts):
    """
    Cleans text data by removing URLs, HTML tags, and non-alphanumeric characters.

    Args:
        texts (list): A list of strings to be cleaned.

    Returns:
        list: A list of cleaned text strings.
    """
    logging.info("Cleaning text data...")
    cleaned_texts = []
    for text in texts:
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        cleaned_texts.append(text.lower().strip())
    logging.debug(f"Cleaned texts: {cleaned_texts}")
    return cleaned_texts

def remove_stopwords(texts):
    """
    Removes stopwords from text data.

    Args:
        texts (list): A list of strings with words to process.

    Returns:
        list: A list of strings with stopwords removed.
    """
    logging.info("Removing stopwords...")
    result = [' '.join([word for word in text.split() if word.lower() not in stop_words]) for text in texts]
    logging.debug(f"Texts after stopword removal: {result}")
    return result

def lemmatize_text(texts):
    """
    Lemmatizes words in text data to their base form.

    Args:
        texts (list): A list of strings to lemmatize.

    Returns:
        list: A list of lemmatized text strings.
    """
    logging.info("Lemmatizing text data...")
    lemmatizer = WordNetLemmatizer()
    result = [' '.join([lemmatizer.lemmatize(word) for word in text.split()]) for text in texts]
    logging.debug(f"Lemmatized texts: {result}")
    return result

def split_into_sentences(texts):
    """
    Splits text into sentences.

    Args:
        texts (list): A list of strings to split into sentences.

    Returns:
        list: A list of lists, where each inner list contains sentences from a text.
    """
    logging.info("Splitting text into sentences...")
    result = [sent_tokenize(text) for text in texts]
    logging.debug(f"Split sentences: {result}")
    return result

# Dataset preparation
def prepare_dataset(input_ids, attention_masks, labels=None):
    """
    Prepares the dataset for model training or evaluation.

    Args:
        input_ids (np.array): Tokenized input IDs.
        attention_masks (np.array): Attention masks for tokenized inputs.
        labels (list, optional): Labels for the dataset. Defaults to None.

    Returns:
        Dataset: A Hugging Face Dataset object.
    """
    logging.info("Preparing dataset...")
    data_dict = {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_masks.tolist()
    }
    if labels is not None:
        data_dict["labels"] = labels  # Use the list directly
    logging.debug(f"Prepared dataset: {data_dict}")
    return Dataset.from_dict(data_dict)

# Model fine-tuning
def fine_tune_model(train_dataset, val_dataset, model_name="bert-base-uncased"):
    """
    Fine-tunes a pretrained BERT model for binary classification.

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        model_name (str): Name of the pretrained model.

    Returns:
        Trainer: A Hugging Face Trainer object with the fine-tuned model.
    """
    logging.info("Initializing model for fine-tuning...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        logging.info("Model initialized successfully.")

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            save_total_limit=2,
            seed=42,
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        logging.info("Starting fine-tuning...")
        trainer.train()
        logging.info("Fine-tuning completed successfully.")
        model.save_pretrained("./fine_tuned_model")
        logging.info("Model saved successfully.")
        return trainer
    except Exception as e:
        logging.error(f"Error during fine-tuning: {e}")
        raise

# Main function
def main():
    """
    Main entry point for the script. Handles preprocessing, tokenization, dataset preparation,
    and model fine-tuning.
    """
    setup_logging()
    logging.info("Starting the script...")

    try:
        # Preprocessing steps
        logging.info("Preprocessing input data...")
        cleaned_texts = clean_text(INPUT_TEXTS)
        texts_no_stopwords = remove_stopwords(cleaned_texts)
        lemmatized_texts = lemmatize_text(texts_no_stopwords)
        split_sentences = split_into_sentences(lemmatized_texts)

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

        # Tokenization
        logging.info("Tokenizing data...")
        tokenized_data = tokenizer(
            lemmatized_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="np"
        )
        input_ids = tokenized_data["input_ids"]
        attention_masks = tokenized_data["attention_mask"]

        # Labels for binary classification
        labels = [0, 1, 0, 1]

        # Splitting data
        logging.info("Splitting data into training and validation sets...")
        train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
            input_ids, attention_masks, labels, test_size=0.2, random_state=42
        )

        train_dataset = prepare_dataset(train_inputs, train_masks, labels=train_labels)
        val_dataset = prepare_dataset(val_inputs, val_masks, labels=val_labels)

        # Fine-tuning
        logging.info("Starting model fine-tuning process...")
        fine_tune_model(train_dataset, val_dataset, model_name=TOKENIZER_MODEL)

        logging.info("Script completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        raise

if __name__ == "__main__":
    main()
