import os
import json
import logging
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Variables
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\LANGGRAPH-3\1\log'
log_filename = "az_cli_faiss.log"
json_folder_path = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\LANGGRAPH-3\1\az-cli-json'
faiss_index_path = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\LANGGRAPH-3\1\faiss_az_cli'




# Initialize logging
def setup_logging():
    """Setup logging configuration with a fixed log filename."""
    try:
        os.makedirs(log_directory, exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(log_directory, log_filename),
            filemode='w'
        )
        logging.debug("✅ Logging setup completed.")
    except Exception as e:
        print(f"⚠️ Failed to setup logging: {e}")

# Load the embedding model
def initialize_embeddings():
    """Initialize the embedding model using Hugging Face."""
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
        logging.debug("✅ Embedding model initialized.")
        print("✅ Embedding model initialized.")
        return embeddings_model
    except Exception as e:
        logging.error(f"❌ Failed to initialize the embedding model: {e}")
        raise

# Process JSON files inside subfolders
def process_and_store_embeddings(json_folder_path, embeddings_model):
    """Recursively process JSON files in subfolders, generate embeddings, and store in FAISS."""
    try:
        docs = []
        metadata_list = []

        # Walk through all subdirectories and find JSON files
        for root, _, files in os.walk(json_folder_path):
            for file_name in files:
                if file_name.endswith(".json"):
                    file_path = os.path.join(root, file_name)

                    # Read the JSON file
                    with open(file_path, 'r', encoding='utf-8') as file:
                        try:
                            data = json.load(file)
                            command = data.get("command", "")
                            description = data.get("description", "")
                            syntax = data.get("syntax", "")
                            examples = "\n".join(data.get("examples", []))

                            # Combine text
                            text = f"Command: {command}\nDescription: {description}\nSyntax: {syntax}\nExamples:\n{examples}"
                            docs.append(text)
                            metadata_list.append({"source": file_path})  # Store full path

                            logging.debug(f"✅ Processed JSON file: {file_path}")
                            print(f"✅ Processed JSON file: {file_path}")

                        except json.JSONDecodeError as e:
                            logging.error(f"❌ Error reading JSON file {file_path}: {e}")
                            print(f"❌ Error reading JSON file {file_path}: {e}")

        # Generate embeddings and store in FAISS
        if docs:
            faiss_db = FAISS.from_texts(docs, embeddings_model, metadatas=metadata_list)
            faiss_db.save_local(faiss_index_path)
            logging.debug(f"✅ Embeddings stored successfully in FAISS at {faiss_index_path}.")
            print(f"✅ Embeddings stored successfully in FAISS at {faiss_index_path}.")
        else:
            logging.error("⚠️ No valid JSON files were processed.")
            print("⚠️ No valid JSON files were processed.")

    except Exception as e:
        logging.error(f"❌ Failed to process and store embeddings: {e}")
        raise

# Main function
def main():
    setup_logging()
    embeddings_model = initialize_embeddings()
    process_and_store_embeddings(json_folder_path, embeddings_model)

if __name__ == "__main__":
    main()
