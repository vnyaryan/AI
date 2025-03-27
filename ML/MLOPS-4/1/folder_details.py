import os
import logging

# Define input folder path
input_folder = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\1\az-cli"

# Define output text file path
output_file = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\1\folders_list.txt"

# Define logging configuration
log_directory = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\1\log"
log_filename = "az-cli-details-1.log"

# Function to set up logging
def setup_logging():
    try:
        os.makedirs(log_directory, exist_ok=True)
        log_file_path = os.path.join(log_directory, log_filename)
        with open(log_file_path, 'w', encoding="utf-8") as log_file:
            log_file.write("# Log file for YAML Processing\n")
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

# Function to list all folders and subfolders
def list_folders(root_folder, output_file):
    try:
        logging.info(f"Processing folder: {root_folder}")
        with open(output_file, "w", encoding="utf-8") as file:
            for dirpath, _, _ in os.walk(root_folder):
                file.write(f"{dirpath}\n")
                logging.info(f"Found folder: {dirpath}")
        logging.info(f"Folder paths saved to: {output_file}")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        print(f"Error occurred: {e}")

# Call logging setup function
setup_logging()

# Run the function
list_folders(input_folder, output_file)
