"""
Script Name: Data Preprocessing for Transformers

Description:
This script preprocesses text data for use with transformer-based models such as BERT.
It performs the following steps:
1. Tokenization: Converts raw text into token IDs using a pre-trained tokenizer.
2. Padding: Ensures that all sequences in a batch are of the same length by adding padding tokens.
3. Attention Masks: Creates binary masks to distinguish real tokens from padding.
4. Special Tokens: Adds [CLS] (classification) and [SEP] (separator) tokens as required.
5. Data Splitting: Splits the tokenized data and attention masks into training and validation sets.

Input:
- A list of raw text sentences to preprocess.
- Pre-trained tokenizer model (e.g., "bert-base-uncased").
- Maximum sequence length to ensure consistent input size.

Output:
- Tokenized input IDs (padded sequences of token IDs).
- Attention masks for the tokenized input.
- Training and validation splits of tokenized input and attention masks.

Logic:
1. Load a pre-trained tokenizer from the Hugging Face library.
2. Tokenize the input text, adding padding and truncating sequences to the defined maximum length.
3. Generate attention masks for the padded sequences.
4. Split the tokenized data and attention masks into training and validation sets.
5. Log detailed debug information about each step for traceability.

Note:
- This script is designed for compatibility with Hugging Face's `transformers` library.
- Ensure the `transformers` library is installed before running the script.




graph TD
    A[Start: Main Function] --> B[Setup Logging]
    B --> C[Load Pre-trained Tokenizer]
    C --> D[Tokenize Input Texts]
    D -->|Generate input_ids| E[Create Attention Masks]
    E -->|Generate attention_masks| F[Split Data into Training and Validation Sets]
    F -->|Training & Validation Sets| G[Log Results]
    G --> H[End]

    %% Tokenization Subgraph
    subgraph Tokenization Workflow
        D --> D1[Add Padding & Truncation]
        D --> D2[Add Special Tokens]
        D --> D3[Return as NumPy Arrays]
    end

    %% Attention Masks Subgraph
    subgraph Attention Masks
        E --> E1[1 for Real Tokens]
        E --> E2[0 for Padding Tokens]
    end

    %% Data Splitting Subgraph
    subgraph Data Splitting
        F --> F1[Split Input IDs]
        F --> F2[Split Attention Masks]
        F --> F3[Set Test Size (Default: 0.2)]
    end


"""

# **Built-in Python Libraries**

import os  
# Provides functions for interacting with the operating system
# Why required:
# - Used for file and directory operations, such as creating the log directory if it doesn't exist.


import logging  

# Offers a flexible framework for generating log messages
# Why required:
# - Used to record debug and info-level logs to track the script's progress and trace errors.

# **Third-Party Libraries**

import numpy as np  

# A library for numerical computations and working with arrays
# Why required:
# - Used to create and manipulate attention masks, ensuring real tokens are marked as 1 and padding as 0.

from transformers import AutoTokenizer 

# Hugging Face library for working with pre-trained models and tokenizers
# Why required:
# - Loads pre-trained tokenizers (e.g., BERT, GPT) to convert text into tokenized input IDs compatible with transformer models.

from sklearn.model_selection import train_test_split  

# A utility from scikit-learn for splitting datasets
# Why required:
# - Used to split tokenized data and attention masks into training and validation sets, ensuring proper evaluation during model training.

# Define constants
INPUT_TEXTS = [
    "I love AI.",
    "Transformers are amazing.",
    "Machine learning is the future.",
    "Natural language processing is fascinating!"
]
TOKENIZER_MODEL = "bert-base-uncased"
MAX_SEQ_LENGTH = 10  

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

def tokenize_texts(texts, tokenizer, max_length):
    """
    Tokenizes the input texts using the specified tokenizer.

    Args:
        texts (list of str): A list of raw text sentences to preprocess.
        tokenizer (transformers.PreTrainedTokenizer): A pre-trained tokenizer from Hugging Face.
        max_length (int): The maximum sequence length for tokenized output.

    Returns:
        dict: A dictionary containing:
            - input_ids (numpy.ndarray): Tokenized and padded sequences of token IDs.
            - token_type_ids (numpy.ndarray): Type IDs for distinguishing multiple sentences (optional).
            - attention_mask (numpy.ndarray): Masks for real tokens and padding.
    
    Logs:
        - Information about tokenization progress and debug-level details of tokenized data.
    """
    logging.info("Starting tokenization...")
    tokenized = tokenizer(
        texts,
        padding="max_length",       # Pad to max_length
        truncation=True,            # Truncate if longer than max_length
        max_length=max_length,      # Maximum sequence length
        return_tensors="np"         # Return as NumPy arrays
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
        - Information about attention mask creation progress and debug-level details of the masks.
    """
    logging.info("Creating attention masks...")
    masks = np.where(input_ids != 0, 1, 0)  # 1 for tokens, 0 for padding
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
        tuple: Four numpy arrays:
            - train_inputs (numpy.ndarray): Training input IDs.
            - val_inputs (numpy.ndarray): Validation input IDs.
            - train_masks (numpy.ndarray): Training attention masks.
            - val_masks (numpy.ndarray): Validation attention masks.
    
    Logs:
        - Information about data splitting progress and debug-level details of the split data.
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
        1. Set up logging.
        2. Load the tokenizer.
        3. Tokenize input texts.
        4. Create attention masks for the tokenized data.
        5. Split the tokenized data into training and validation sets.
        6. Log the results.

    Logs:
        - High-level progress and results of each preprocessing step.
    """
    setup_logging()

    logging.info("Starting the data preprocessing script...")
    
    # Load the tokenizer
    logging.info("Loading the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    logging.info("Tokenizer loaded successfully.")



    # Step 1, 2, 3: Tokenize the input texts (includes padding and special tokens)
    tokenized_data = tokenize_texts(INPUT_TEXTS, tokenizer, MAX_SEQ_LENGTH)
    input_ids = tokenized_data["input_ids"]

    # Step 4: Create attention masks
    attention_masks = create_attention_masks(input_ids)

    # Step 5: Split data into training and validation sets
    train_inputs, val_inputs, train_masks, val_masks = split_data(input_ids, attention_masks)

    logging.info("Data preprocessing completed successfully.")
    logging.info("Results:")
    logging.debug(f"Training Inputs: {train_inputs}")
    logging.debug(f"Validation Inputs: {val_inputs}")
    logging.debug(f"Training Masks: {train_masks}")
    logging.debug(f"Validation Masks: {val_masks}")

if __name__ == "__main__":
    main()
