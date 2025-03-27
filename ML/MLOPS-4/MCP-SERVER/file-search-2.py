import sqlite3
import pandas as pd
import os
import logging

# Log configuration
log_directory = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\2\log"
log_filename = "file_search_filedetails.log"

# Setup logging function
def setup_logging():
    try:
        os.makedirs(log_directory, exist_ok=True)
        log_file_path = os.path.join(log_directory, log_filename)
        with open(log_file_path, 'w', encoding="utf-8") as log_file:
            log_file.write("# Log file for File Search\n")
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

# Search function using pandas
def search_files(db_path, keyword):
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT path, content FROM files WHERE content MATCH ?"
        df = pd.read_sql_query(query, conn, params=(keyword,))
        conn.close()

        if not df.empty:
            logging.info(f"Keyword '{keyword}' matched {len(df)} file(s).")
            print(f"\nFound {len(df)} file(s) containing '{keyword}':\n")
            print(df[['path']].to_string(index=False))
        else:
            logging.info(f"No matches found for keyword '{keyword}'.")
            print(f"No matches found for '{keyword}'.")

    except Exception as e:
        logging.error(f"Error during search: {e}")
        print(f"An error occurred during search: {e}")

# Main execution
if __name__ == "__main__":
    setup_logging()
    db_path = "file_index.db"
    keyword = input("Enter keyword to search: ")
    logging.info(f"Search initiated for keyword: '{keyword}'")
    search_files(db_path, keyword)
