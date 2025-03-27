import os
import logging

# Define input and output file paths
input_file = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\1\folders_list.txt"
output_file = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\1\yaml_files_list.txt"

# Define logging configuration
log_directory = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\1\log"
log_filename = "yaml_files_processing.log"

# Function to set up logging
def setup_logging():
    try:
        os.makedirs(log_directory, exist_ok=True)
        log_file_path = os.path.join(log_directory, log_filename)
        with open(log_file_path, 'w', encoding="utf-8") as log_file:
            log_file.write("# Log file for YAML Files Processing\n")
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

# Function to process folders and find YAML files
def process_folders(input_file, output_file):
    separator = "=" * 79  # 79-character separator
    try:
        with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
            for line in infile:
                folder_path = line.strip()
                if os.path.isdir(folder_path):
                    yaml_files = [f for f in os.listdir(folder_path) if f.endswith(('.yml', '.yaml'))]
                    
                    outfile.write(f"{separator}\n")  # Add separator before each folder
                    outfile.write(f"{folder_path}\n")
                    logging.info(f"Processing folder: {folder_path}")

                    if yaml_files:
                        for yaml_file in yaml_files:
                            outfile.write(f"    {yaml_file}\n")
                            logging.info(f"Found YAML file: {yaml_file} in {folder_path}")
                    else:
                        outfile.write("    No YAML files found\n")
                        logging.info(f"No YAML files found in {folder_path}")
                    
                    outfile.write(f"{separator}\n")  # Add separator after each folder

        logging.info(f"YAML file list saved to: {output_file}")
        print(f"YAML file list saved to: {output_file}")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        print(f"Error occurred: {e}")

# Call logging setup function
setup_logging()

# Run the function
process_folders(input_file, output_file)
