import os
import re
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

# Constants
INPUT_TEXTS = [
    "I love AI.",
    "Transformers are amazing.",
    "Machine learning is the future.",
    "Natural language processing is fascinating!"
]
TOKENIZER_MODEL = "bert-base-uncased"
MAX_SEQ_LENGTH = 10
stop_words = set(stopwords.words('english'))

# Setup logging
def setup_logging():
    """
    Sets up logging for the application.

    Creates the log directory if it does not exist and configures the logging settings 
    to save logs in the specified file with detailed debug information.

    Raises:
        Exception: If there's an error while setting up logging.
    """
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-1\2'
        log_filename = 'mlops-4.log'

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
    logging.info("Removing stopwords...")
    result = [' '.join([word for word in text.split() if word.lower() not in stop_words]) for text in texts]
    logging.debug(f"Texts after stopword removal: {result}")
    return result

def lemmatize_text(texts):
    logging.info("Lemmatizing text data...")
    lemmatizer = WordNetLemmatizer()
    result = [' '.join([lemmatizer.lemmatize(word) for word in text.split()]) for text in texts]
    logging.debug(f"Lemmatized texts: {result}")
    return result

def split_into_sentences(texts):
    logging.info("Splitting text into sentences...")
    result = [sent_tokenize(text) for text in texts]
    logging.debug(f"Split sentences: {result}")
    return result

# Dataset preparation
def prepare_dataset(input_ids, attention_masks, labels=None):
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
