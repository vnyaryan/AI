import os
import logging

# Define log configuration
log_directory = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\2\log"
log_filename = "file_search_content.log"

# Setup logging
def setup_logging():
    try:
        os.makedirs(log_directory, exist_ok=True)
        log_file_path = os.path.join(log_directory, log_filename)
        with open(log_file_path, 'w', encoding="utf-8") as log_file:
            log_file.write("# Log file for Keyword Line Search\n")
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

# Function to search keyword in file lines
def search_keyword_in_file(file_path, keyword):
    matched_lines = 0
    try:
        logging.info(f"Opening file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if keyword.lower() in line.lower():
                    print(line.strip())  # Output line exactly as in file
                    matched_lines += 1

        logging.info(f"Keyword '{keyword}' found in {matched_lines} line(s).")

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        print(f"Error: File not found: {file_path}")
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        print(f"Error reading file: {e}")

# Main execution
if __name__ == "__main__":
    setup_logging()
    file_path = input("Enter the full file path: ").strip()
    keyword = input("Enter keyword to search: ").strip()
    logging.info(f"Search started for keyword: '{keyword}' in file: {file_path}")
    search_keyword_in_file(file_path, keyword)
