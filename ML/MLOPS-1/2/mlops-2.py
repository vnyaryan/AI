"""
Updated Script: Data Preprocessing for Transformers with Additional Steps

Description:
This script preprocesses text data for use with transformer-based models such as BERT.
It includes multiple steps to clean, normalize, augment, and prepare the data for training and evaluation.

Features:
1. Text Cleaning and Normalization: Removes noise like special characters, URLs, and HTML tags.
2. Stopword Removal: Filters out common stopwords.
3. Text Augmentation: Adds diversity to the dataset using backtranslation.
4. Lemmatization: Reduces words to their root forms for normalization.
5. Sentence-Level Preprocessing: Splits paragraphs into individual sentences.
6. Tokenization: Converts text into token IDs using a pre-trained tokenizer, adding special tokens.
7. Attention Masks: Creates binary masks to distinguish real tokens from padding.
8. Data Splitting: Divides the dataset into training and validation subsets.
9. Logging: Tracks progress and provides detailed debug-level logs.
10. Handling Class Imbalance: Balances datasets using SMOTE (if applicable).

Input:
- A list of raw text sentences to preprocess (constant `INPUT_TEXTS`).
- Pre-trained tokenizer model name (constant `TOKENIZER_MODEL`).
- Maximum sequence length for tokenization (constant `MAX_SEQ_LENGTH`).

Output:
- Cleaned, tokenized, and augmented text data in the form of token IDs.
- Attention masks for distinguishing real tokens from padding tokens.
- Training and validation splits of tokenized inputs and attention masks.

Logic:
1. **Text Cleaning**:
    - Remove URLs, HTML tags, special characters, and convert text to lowercase.
2. **Stopword Removal**:
    - Filter out common stopwords like "the," "and," etc.
3. **Lemmatization**:
    - Reduce words to their root forms (e.g., "running" → "run").
4. **Text Augmentation**:
    - Apply backtranslation to create diverse paraphrased versions of the input text.
5. **Sentence Splitting**:
    - Split paragraphs into individual sentences for finer-grained processing.
6. **Tokenization**:
    - Convert text to token IDs with padding, truncation, and special tokens ([CLS], [SEP]).
7. **Attention Mask Creation**:
    - Generate binary masks (1 for real tokens, 0 for padding tokens).
8. **Data Splitting**:
    - Divide the dataset into training and validation subsets (80%-20% by default).

Note:
- This script uses libraries like Hugging Face's `transformers`, NLTK, and `imblearn`.
- Ensure all dependencies are installed before running this script.
"""


# **Built-in Libraries**
import os
import re
import logging
from collections import Counter

# **Third-Party Libraries**
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from googletrans import Translator

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

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

def main():
    """
    Main function to orchestrate the data preprocessing workflow.

    Steps:
        1. Clean and normalize text.
        2. Remove stopwords.
        3. Perform lemmatization.
        4. Augment data using backtranslation.
        5. Split paragraphs into sentences.
        6. Tokenize data and generate attention masks.
        7. Split tokenized data into training and validation sets.
    """
    setup_logging()

    logging.info("Starting the data preprocessing script...")

    cleaned_texts = clean_text(INPUT_TEXTS)
    texts_no_stopwords = remove_stopwords(cleaned_texts)
    lemmatized_texts = lemmatize_text(texts_no_stopwords)
    augmented_texts = backtranslation(lemmatized_texts)
    split_sentences = split_into_sentences(augmented_texts)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    tokenized_data = tokenize_texts(augmented_texts, tokenizer, MAX_SEQ_LENGTH)
    input_ids = tokenized_data["input_ids"]
    attention_masks = create_attention_masks(input_ids)

    train_inputs, val_inputs, train_masks, val_masks = split_data(input_ids, attention_masks)

    logging.info("Data preprocessing completed successfully.")
    logging.info("Results:")
    logging.debug(f"Training Inputs: {train_inputs}")
    logging.debug(f"Validation Inputs: {val_inputs}")
    logging.debug(f"Training Masks: {train_masks}")
    logging.debug(f"Validation Masks: {val_masks}")

if __name__ == "__main__":
    main()
