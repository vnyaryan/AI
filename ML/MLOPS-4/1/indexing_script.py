# indexing_script.py
import os
import logging
import meilisearch

# Define logging configuration
log_directory = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\1\log"
log_filename = "indexing_script.log"

# Function to set up logging
def setup_logging():
    try:
        os.makedirs(log_directory, exist_ok=True)
        log_file_path = os.path.join(log_directory, log_filename)
        with open(log_file_path, 'w', encoding="utf-8") as log_file:
            log_file.write("# Log file for File Indexing\n")
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=log_file_path,
            filemode='a'
        )
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        raise

# Function to index files
def index_files(root_dir):
    try:
        documents = []
        for foldername, subfolders, filenames in os.walk(root_dir):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    doc = {
                        'path': file_path,
                        'name': filename,
                        'content': content
                    }
                    documents.append(doc)
                    logging.info(f"Indexed file: {file_path}")
                except Exception as file_error:
                    logging.error(f"Skipping file {file_path}: {file_error}")

        if documents:
            index.add_documents(documents)
            logging.info(f"Indexed {len(documents)} files to Meilisearch.")
        else:
            logging.warning("No files indexed.")
    except Exception as e:
        logging.error(f"Error during indexing: {e}")

# Main Execution
if __name__ == "__main__":
    setup_logging()
    try:
        # Connect to Meilisearch
        client = meilisearch.Client("http://127.0.0.1:7700", "mySecretKey")
        index_name = "files"

        try:
            client.create_index(index_name, {'primaryKey': 'path'})
            logging.info(f"Index '{index_name}' created.")
        except meilisearch.errors.MeiliSearchApiError:
            logging.info(f"Index '{index_name}' already exists.")

        index = client.index(index_name)

        # Directory to index
        directory_to_index = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\1\az-cli-json\account"
        logging.info(f"Starting indexing for directory: {directory_to_index}")
        index_files(directory_to_index)

    except Exception as connection_error:
        logging.error(f"Failed to connect or index files: {connection_error}")
