import os
import yaml
import logging

# Constants for logging and output
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\7\facts\logs'
log_filename = 'cli_processing.log'
output_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\7\facts\commands'  # Directory to save individual command files
file_list = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\7\facts\filecheck.txt'  # File with paths of all YAML files

# Function to set up logging
def setup_logging():
    os.makedirs(log_directory, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=os.path.join(log_directory, log_filename),
                        filemode='w')
    logging.debug("Logging setup completed.")

# Function to process YAML file and extract UID, Syntax, Parameters, and Examples
def extract_details(yaml_file_path):
    try:
        with open(yaml_file_path, 'r', encoding='utf-8') as file:
            content = yaml.safe_load(file)
            if content:
                uid = content.get('uid', 'Unknown UID')
                name = content.get('name', 'Unknown Name')
                summary = content.get('summary', 'No summary available')

                # Extract Syntax and Parameters
                syntax = ""
                parameters = ""
                examples = ""

                if 'directCommands' in content:
                    for command in content['directCommands']:
                        syntax = command.get('syntax', 'No syntax available')
                        required_params = ", ".join([param['name'] for param in command.get('requiredParameters', [])]) or "No required parameters"
                        optional_params = ", ".join([param['name'] for param in command.get('optionalParameters', [])]) or "No optional parameters"
                        parameters = f"Required: {required_params}, Optional: {optional_params}"

                        # Extract Examples
                        example_list = []
                        if 'examples' in command:
                            for example in command['examples']:
                                example_summary = example.get('summary', 'No example summary')
                                example_syntax = example.get('syntax', 'No example syntax')
                                example_list.append(f"Example: {example_summary}, Syntax: {example_syntax}")
                        examples = "\n".join(example_list) if example_list else "No examples provided"
                
                return uid, f"UID: {uid}\nName: {name}\nSummary: {summary}\nSyntax: {syntax}\nParameters: {parameters}\nExamples: {examples}"
            else:
                return None, f"UID: Unknown, Name: Unknown, Summary: Empty file: {yaml_file_path}"
    except Exception as e:
        logging.error(f"Error processing {yaml_file_path}: {e}")
        return None, f"Error processing {yaml_file_path}: {e}"

# Function to save details in separate files
def save_details_to_file(uid, details):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        output_file_path = os.path.join(output_directory, f"{uid}.txt")
        
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(details)
        logging.debug(f"Saved details to file: {output_file_path}")
    except Exception as e:
        logging.error(f"Error saving file {uid}: {e}")

# Function to read file paths and process each YAML
def process_all_files(file_list_path):
    try:
        with open(file_list_path, 'r', encoding='utf-8') as file_list_file:
            yaml_files = file_list_file.readlines()
            logging.info(f"Found {len(yaml_files)} YAML files to process.")

            for yaml_file in yaml_files:
                yaml_file = yaml_file.strip()
                if os.path.exists(yaml_file):
                    uid, details = extract_details(yaml_file)
                    if uid:
                        save_details_to_file(uid, details)
                    logging.debug(f"Processed and extracted details: {yaml_file}")
                else:
                    logging.warning(f"File not found: {yaml_file}")

            logging.info("Processing completed.")
    except Exception as e:
        logging.error(f"Failed to process files: {e}")

# Main function to execute the processing
def main():
    setup_logging()
    logging.info("Starting YAML file processing...")
    process_all_files(file_list)
    logging.info("Processing completed.")

if __name__ == "__main__":
    main()
