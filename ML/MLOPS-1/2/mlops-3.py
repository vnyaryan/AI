"""
Script Overview:
----------------
This script is designed to preprocess text data, augment it using backtranslation, and fine-tune a pre-trained 
transformer model for classification tasks. Below is a detailed explanation of the script's inputs, outputs, 
and its step-by-step logic.

Input:
    - A list of raw text sentences (defined in the `INPUT_TEXTS` constant).
    - Pre-trained model name for tokenization and fine-tuning (e.g., "bert-base-uncased").
    - Constants and configurations like maximum sequence length and logging settings.

Output:
    - Cleaned, tokenized, and preprocessed text data.
    - Augmented datasets ready for model training.
    - A fine-tuned transformer model saved to the specified output directory.

Step-by-Step Logic:
-------------------
1. **Setup Logging**: Initializes logging to record the progress and debug information.
2. **Clean and Normalize Text**: Removes special characters, URLs, HTML tags, and converts text to lowercase.
3. **Stopword Removal**: Eliminates common stopwords (e.g., "and", "the") to focus on meaningful content.
4. **Lemmatization**: Reduces words to their base forms (e.g., "running" -> "run").
5. **Backtranslation for Augmentation**: Enhances the dataset by translating text to another language and back.
6. **Split into Sentences**: Divides paragraphs into individual sentences for granular processing.
7. **Tokenization**: Uses a pre-trained tokenizer to convert text into numerical tokens for model input.
8. **Attention Mask Generation**: Creates binary masks to distinguish between real tokens and padding.
9. **Data Splitting**: Splits tokenized data into training and validation sets.
10. **Dataset Preparation**: Formats data into a Hugging Face Dataset object for training.
11. **Model Fine-Tuning**: Fine-tunes a transformer model using Hugging Face's Trainer.
12. **Save Model**: Saves the fine-tuned model for future inference.

Execution:
----------
- This script can be executed directly, and it will process the provided input data and fine-tune the model 
  automatically.

Dependencies:
-------------
- Install required Python packages using pip: 
  `pip install transformers scikit-learn imbalanced-learn googletrans datasets nltk`.

Notes:
------
- Ensure NLTK resources (`stopwords`, `wordnet`, `punkt`) are downloaded before running the script.
- Logging details and intermediate results are saved in the specified log file for troubleshooting.
"""



import os  # For creating directories and managing file paths.
import re  # For performing regular expression operations to clean and process text.
import logging  # For logging messages and errors during script execution.
from collections import Counter  # For counting occurrences of elements (used for analyzing data distributions).
import numpy as np  # For numerical computations and array manipulations.
import nltk  # For natural language processing tasks like tokenization and lemmatization.
from nltk.corpus import stopwords  # For accessing a collection of common stopwords.
from nltk.stem import WordNetLemmatizer  # For lemmatizing words to their base forms.
from nltk.tokenize import sent_tokenize  # For splitting paragraphs into individual sentences.
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# - `AutoTokenizer`: For tokenizing text using pre-trained tokenizer models.
# - `AutoModelForSequenceClassification`: For loading a pre-trained transformer model for classification.
# - `Trainer`: For training and fine-tuning the transformer model.
# - `TrainingArguments`: For configuring training hyperparameters.

from sklearn.model_selection import train_test_split  # For splitting data into training and validation sets.
from imblearn.over_sampling import SMOTE  # For oversampling minority classes to handle class imbalance.
from googletrans import Translator  # For performing backtranslation for data augmentation.
from datasets import Dataset  # For creating and managing datasets compatible with Hugging Face models.

# Download necessary NLTK resources
nltk.download('stopwords')  # Downloads the stopwords dataset for filtering common stopwords.
nltk.download('wordnet')  # Downloads the WordNet lexical database for lemmatization.
nltk.download('punkt')  # Downloads the Punkt tokenizer model for splitting text into sentences.
nltk.download('punkt_tab')


# **Constants**
INPUT_TEXTS = [
    "I love AI.",
    "Transformers are amazing.",
    "Machine learning is the future.",
    "Natural language processing is fascinating!"
]
TOKENIZER_MODEL = "bert-base-uncased"
MAX_SEQ_LENGTH = 10
stop_words = set(stopwords.words('english'))

def setup_logging():
    """
    Sets up logging for the application.

    Creates the log directory if it does not exist and configures the logging settings 
    to save logs in the specified file with detailed debug information.

    Raises:
        Exception: If there's an error while setting up logging.
    """
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-1\1'
        log_filename = 'mlops-1.log'

        os.makedirs(log_directory, exist_ok=True)

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='w')
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        raise

def clean_text(texts):
    """
    Cleans and normalizes text by removing special characters, URLs, and HTML tags.

    Args:
        texts (list of str): A list of raw text sentences.

    Returns:
        list of str: Cleaned and normalized text.

    Logs:
        - Information about text cleaning progress.
        - Debug-level details of cleaned texts.
    """
    logging.info("Cleaning and normalizing text data...")
    cleaned_texts = []
    for text in texts:
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"<.*?>", "", text)    # Remove HTML tags
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
        cleaned_texts.append(text.lower().strip())  # Convert to lowercase and trim
    logging.debug(f"Cleaned Texts: {cleaned_texts}")
    return cleaned_texts

def remove_stopwords(texts):
    """
    Removes stopwords from the input text.

    Args:
        texts (list of str): A list of raw text sentences.

    Returns:
        list of str: Texts with stopwords removed.

    Logs:
        - Information about stopword removal progress.
        - Debug-level details of texts after stopword removal.
    """
    logging.info("Removing stopwords from input texts...")
    cleaned_texts = [' '.join([word for word in text.split() if word.lower() not in stop_words]) for text in texts]
    logging.debug(f"Texts after stopword removal: {cleaned_texts}")
    return cleaned_texts

def lemmatize_text(texts):
    """
    Applies lemmatization to reduce words to their base or root forms.

    Args:
        texts (list of str): A list of tokenized text sentences.

    Returns:
        list of str: Lemmatized text.

    Logs:
        - Information about lemmatization progress.
        - Debug-level details of lemmatized texts.
    """
    logging.info("Applying lemmatization...")
    lemmatizer = WordNetLemmatizer()
    lemmatized_texts = [' '.join([lemmatizer.lemmatize(word) for word in text.split()]) for text in texts]
    logging.debug(f"Lemmatized Texts: {lemmatized_texts}")
    return lemmatized_texts

def backtranslation(texts, src_lang='en', target_lang='fr'):
    """
    Performs backtranslation for text augmentation.

    Args:
        texts (list of str): A list of raw text sentences.
        src_lang (str): Source language.
        target_lang (str): Target language for translation.

    Returns:
        list of str: Backtranslated texts.

    Logs:
        - Information about backtranslation progress.
        - Warnings for failed translations.
        - Debug-level details of augmented texts.
    """
    logging.info("Performing backtranslation for text augmentation...")
    translator = Translator()
    augmented_texts = []
    for text in texts:
        try:
            translated = translator.translate(text, src=src_lang, dest=target_lang).text
            backtranslated = translator.translate(translated, src=target_lang, dest=src_lang).text
            augmented_texts.append(backtranslated)
        except Exception as e:
            logging.warning(f"Failed to backtranslate text: {text}. Error: {e}")
            augmented_texts.append(text)  # Use original text if translation fails
    logging.debug(f"Texts after backtranslation: {augmented_texts}")
    return augmented_texts

def split_into_sentences(texts):
    """
    Splits paragraphs into individual sentences.

    Args:
        texts (list of str): A list of paragraphs.

    Returns:
        list of list of str: A nested list of sentences for each paragraph.

    Logs:
        - Information about sentence splitting progress.
        - Debug-level details of split sentences.
    """
    logging.info("Splitting text into sentences...")
    sentences = [sent_tokenize(text) for text in texts]
    logging.debug(f"Split Sentences: {sentences}")
    return sentences

def tokenize_texts(texts, tokenizer, max_length):
    """
    Tokenizes the input texts using the specified tokenizer.

    Args:
        texts (list of str): A list of text sentences to preprocess.
        tokenizer (transformers.PreTrainedTokenizer): A pre-trained tokenizer from Hugging Face.
        max_length (int): The maximum sequence length for tokenized output.

    Returns:
        dict: Tokenized data containing input_ids, attention masks, and token type IDs.

    Logs:
        - Information about tokenization progress.
        - Debug-level details of tokenized data.
    """
    logging.info("Starting tokenization...")
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_tensors="np"
    )
    logging.info("Tokenization completed.")
    logging.debug(f"Tokenized Data: {tokenized}")
    return tokenized

def create_attention_masks(input_ids):
    """
    Creates attention masks to differentiate real tokens from padding tokens.

    Args:
        input_ids (numpy.ndarray): Tokenized input IDs with padding.

    Returns:
        numpy.ndarray: A binary mask array (1 for real tokens, 0 for padding).

    Logs:
        - Information about attention mask creation progress.
        - Debug-level details of the masks.
    """
    logging.info("Creating attention masks...")
    masks = np.where(input_ids != 0, 1, 0)
    logging.info("Attention masks created.")
    logging.debug(f"Attention Masks: {masks}")
    return masks

def split_data(input_ids, attention_masks, test_size=0.2):
    """
    Splits the input data into training and validation sets.

    Args:
        input_ids (numpy.ndarray): Tokenized input IDs.
        attention_masks (numpy.ndarray): Corresponding attention masks for the input IDs.
        test_size (float): Proportion of the data to include in the validation split.

    Returns:
        tuple: Four arrays for training and validation inputs and masks.

    Logs:
        - Information about data splitting progress.
        - Debug-level details of split datasets.
    """
    logging.info("Splitting data into training and validation sets...")
    train_inputs, val_inputs, train_masks, val_masks = train_test_split(
        input_ids, attention_masks, test_size=test_size, random_state=42
    )
    logging.info("Data splitting completed.")
    logging.debug(f"Training Input IDs: {train_inputs}")
    logging.debug(f"Validation Input IDs: {val_inputs}")
    logging.debug(f"Training Attention Masks: {train_masks}")
    logging.debug(f"Validation Attention Masks: {val_masks}")
    return train_inputs, val_inputs, train_masks, val_masks

def prepare_dataset(input_ids, attention_masks, labels=None):
    """
    Prepares a Hugging Face Dataset object from tokenized inputs and labels.

    Args:
        input_ids (numpy.ndarray): Tokenized input IDs.
        attention_masks (numpy.ndarray): Attention masks.
        labels (numpy.ndarray, optional): Labels for supervised learning.

    Returns:
        Dataset: A Hugging Face Dataset object.
    """
    data_dict = {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_masks.tolist()
    }
    if labels is not None:
        data_dict["labels"] = labels.tolist()
    return Dataset.from_dict(data_dict)

def fine_tune_model(train_dataset, val_dataset, model_name="bert-base-uncased"):
    """
    Fine-tunes a pre-trained transformer model using Hugging Face's Trainer.

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        model_name (str): The pre-trained model to use.

    Returns:
        Trainer: The Trainer object after training.
    """
    logging.info("Loading the pre-trained model for fine-tuning...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
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
        eval_dataset=val_dataset,
        tokenizer=AutoTokenizer.from_pretrained(model_name),
    )

    logging.info("Starting model fine-tuning...")
    trainer.train()
    logging.info("Model fine-tuning completed.")

    logging.info("Saving the fine-tuned model...")
    model.save_pretrained("./fine_tuned_model")
    logging.info("Fine-tuned model saved successfully.")

    return trainer

def main():
    """
    Main function to orchestrate the data preprocessing and model training workflow.

    Steps:
        1. Clean and normalize text.
        2. Remove stopwords.
        3. Perform lemmatization.
        4. Augment data using backtranslation.
        5. Split paragraphs into sentences.
        6. Tokenize data and generate attention masks.
        7. Split tokenized data into training and validation sets.
        8. Prepare datasets for Hugging Face Trainer.
        9. Fine-tune the model using the prepared datasets.
    """
    # Set up logging to record detailed information about script execution.
    setup_logging()

    # Log the start of the data preprocessing and model training workflow.
    logging.info("Starting the data preprocessing script...")

    # Step 1: Clean and normalize the input text data by removing URLs, HTML tags, and special characters.
    cleaned_texts = clean_text(INPUT_TEXTS)

    # Step 2: Remove common stopwords (e.g., "and", "the") from the cleaned text data.
    texts_no_stopwords = remove_stopwords(cleaned_texts)

    # Step 3: Apply lemmatization to convert words to their base forms (e.g., "running" -> "run").
    lemmatized_texts = lemmatize_text(texts_no_stopwords)

    # Step 4: Perform backtranslation (translate to another language and back) to augment the dataset.
    augmented_texts = backtranslation(lemmatized_texts)

    # Step 5: Split the augmented text into individual sentences for more granular processing.
    split_sentences = split_into_sentences(augmented_texts)

    # Step 6: Initialize a pre-trained tokenizer for the transformer model.
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    # Step 7: Tokenize the augmented text data and pad/truncate it to the maximum sequence length.
    tokenized_data = tokenize_texts(augmented_texts, tokenizer, MAX_SEQ_LENGTH)

    # Extract tokenized input IDs and attention masks from the tokenized data.
    input_ids = tokenized_data["input_ids"]
    attention_masks = create_attention_masks(input_ids)

    # Step 8: Split the tokenized data into training and validation datasets.
    train_inputs, val_inputs, train_masks, val_masks = split_data(input_ids, attention_masks)

    # Log the preparation of datasets for training.
    logging.info("Preparing datasets for Hugging Face Trainer...")

    # Step 9: Create Hugging Face Dataset objects for training and validation data.
    train_dataset = prepare_dataset(train_inputs, train_masks)
    val_dataset = prepare_dataset(val_inputs, val_masks)

    # Log the start of the fine-tuning process.
    logging.info("Initiating fine-tuning process...")

    # Step 10: Fine-tune the transformer model using the prepared datasets.
    fine_tune_model(train_dataset, val_dataset, model_name=TOKENIZER_MODEL)

    # Log the successful completion of the data preprocessing and model training workflow.
    logging.info("Data preprocessing and model training completed successfully.")

# Entry point of the script to ensure the `main` function is executed when the script runs.
if __name__ == "__main__":
    main()
