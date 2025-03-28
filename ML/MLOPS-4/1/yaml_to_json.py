import os
import yaml
import json
import logging

# Define paths
yaml_list_file = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\1\yaml_files_list.txt"
input_root = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\1\az-cli"
output_root = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\1\az-cli-json"
log_directory = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\1\log"
log_filename = "yaml_to_json_conversion.log"

def setup_logging():
    """Sets up logging to capture script execution details."""
    os.makedirs(log_directory, exist_ok=True)
    log_file_path = os.path.join(log_directory, log_filename)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file_path,
        filemode="w",
    )
    logging.debug("Logging setup completed.")

def parse_yaml_list(yaml_list_file):
    """Parses yaml_files_list.txt to extract folder paths and corresponding YAML files."""
    folder_yaml_mapping = {}
    with open(yaml_list_file, "r", encoding="utf-8") as file:
        current_folder = None
        for line in file:
            line = line.strip()
            if line.startswith("C:\\"):  # Folder path
                current_folder = line
                folder_yaml_mapping[current_folder] = []
            elif line.endswith(".yml") or line.endswith(".yaml"):  # YAML file name
                if current_folder:
                    folder_yaml_mapping[current_folder].append(line)
    
    return folder_yaml_mapping

def convert_yaml_to_json(yaml_path, json_path):
    """Converts a YAML file to JSON with a structured format for FASSIS."""
    try:
        if not os.path.exists(yaml_path):
            json_data = { "error": "YAML FILE NOT FOUND" }
            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump(json_data, json_file, indent=4, ensure_ascii=False)
            logging.warning(f"YAML file not found, created error JSON: {json_path}")
            return

        with open(yaml_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if not data:
            json_data = { "error": "YAML FILE NOT PROCESSED PROPERLY" }
            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump(json_data, json_file, indent=4, ensure_ascii=False)
            logging.warning(f"YAML file is empty or invalid, created error JSON: {json_path}")
            return

        # Structured JSON format
        json_data = {
            "command_group": data.get("name", ""),
            "description": data.get("summary", ""),
            "status": data.get("status", ""),
            "commands": []
        }

        if "directCommands" in data:
            for command in data["directCommands"]:
                cmd_details = {
                    "name": command.get("name", ""),
                    "summary": command.get("summary", ""),
                    "syntax": command.get("syntax", ""),
                    "required_parameters": [
                        param["name"] for param in command.get("requiredParameters", [])
                    ] if "requiredParameters" in command else [],
                    "optional_parameters": [
                        param["name"] for param in command.get("optionalParameters", [])
                    ] if "optionalParameters" in command else [],
                    "examples": [
                        example["syntax"] for example in command.get("examples", [])
                    ] if "examples" in command else []
                }
                json_data["commands"].append(cmd_details)

        # Save JSON file
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)

        logging.info(f"Converted YAML to JSON: {yaml_path} -> {json_path}")

    except Exception as e:
        json_data = { "error": "YAML FILE NOT PROCESSED PROPERLY" }
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)
        logging.error(f"Error processing {yaml_path}, created error JSON: {json_path}. Error: {e}")

def process_yaml_files(yaml_list_file, input_root, output_root):
    """Processes all YAML files listed in yaml_files_list.txt and converts them to JSON."""
    os.makedirs(output_root, exist_ok=True)
    folder_yaml_mapping = parse_yaml_list(yaml_list_file)

    for folder_path, yaml_files in folder_yaml_mapping.items():
        # Preserve subfolder structure inside output_root
        relative_folder_path = os.path.relpath(folder_path, input_root)
        output_folder = os.path.join(output_root, relative_folder_path)

        os.makedirs(output_folder, exist_ok=True)  # Create folder structure

        for yaml_file in yaml_files:
            yaml_path = os.path.join(folder_path, yaml_file)
            json_file_name = yaml_file.replace(".yaml", ".json").replace(".yml", ".json")
            json_path = os.path.join(output_folder, json_file_name)

            convert_yaml_to_json(yaml_path, json_path)  # Process each YAML file

    print(f"All YAML files have been converted to JSON in {output_root}")

# Setup logging
setup_logging()

# Process YAML files while maintaining subfolder structure
process_yaml_files(yaml_list_file, input_root, output_root)
