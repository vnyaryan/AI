import os
import sqlite3
import logging

# Define log configuration
log_directory = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\2\log"
log_filename = "filesearch_indexing.log"

# Setup logging function
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

# Indexing function
def index_files_to_sqlite(directory_to_index, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create FTS5 table
        cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS files USING fts5(path, content)
        ''')
        logging.info("FTS5 table created or already exists.")

        file_count = 0
        for root, dirs, files in os.walk(directory_to_index):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    cursor.execute('INSERT INTO files (path, content) VALUES (?, ?)', (file_path, content))
                    file_count += 1
                    logging.info(f"Indexed file: {file_path}")
                except Exception as file_error:
                    logging.error(f"Skipping {file_path}: {file_error}")

        conn.commit()
        conn.close()
        logging.info(f"Indexing complete. Total files indexed: {file_count}")
        print("Indexing complete.")

    except Exception as db_error:
        logging.error(f"Database error occurred: {db_error}")
        print(f"Database error: {db_error}")

# Main execution
if __name__ == "__main__":
    setup_logging()
    db_path = "file_index.db"
    directory_to_index = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\2\az-cli-json"
    logging.info(f"Starting indexing for directory: {directory_to_index}")
    index_files_to_sqlite(directory_to_index, db_path)
