import os
import logging

# Define input paths
yaml_root = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\1\az-cli"
json_root = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\1\az-cli-json"
log_directory = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-4\1\log"
log_filename = "yaml_json_comparison_log.log"

def setup_logging():
    """Sets up logging for folder and file verification."""
    os.makedirs(log_directory, exist_ok=True)
    log_file_path = os.path.join(log_directory, log_filename)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file_path,
        filemode="w",
    )
    logging.debug("Logging setup completed.")

def get_all_folders(root_folder):
    """Returns a sorted list of all folder paths inside a given root directory."""
    folder_list = []
    for dirpath, _, _ in os.walk(root_folder):
        folder_list.append(os.path.relpath(dirpath, root_folder))
    return sorted(folder_list)

def get_yaml_json_files(root_folder, file_extensions):
    """Returns a dictionary mapping folders to YAML or JSON files."""
    file_structure = {}
    for dirpath, _, filenames in os.walk(root_folder):
        relative_path = os.path.relpath(dirpath, root_folder)
        files = [f for f in filenames if f.endswith(file_extensions)]
        if files:
            file_structure[relative_path] = files
    return file_structure

def compare_folders(yaml_root, json_root):
    """Compares folder structures between az-cli and az-cli-json."""
    yaml_folders = get_all_folders(yaml_root)
    json_folders = get_all_folders(json_root)

    missing_folders = []

    for folder in yaml_folders:
        if folder not in json_folders:
            missing_folders.append(f"Missing Folder: {os.path.join(json_root, folder)}")
            logging.warning(f"Missing Folder: {os.path.join(json_root, folder)}")

    # Print Summary
    print(f"\n✅ Total Folders in az-cli: {len(yaml_folders)}")
    print(f"✅ Total Folders in az-cli-json: {len(json_folders)}")

    if missing_folders:
        print(f"\n❌ Mismatch Found: {len(missing_folders)} folders missing in az-cli-json!")
        print("Check `yaml_json_comparison_log.log` for details.")
    else:
        print("\n✅ Folder Structure Matches: No missing folders detected.")

def compare_yaml_json_files(yaml_root, json_root):
    """Compares YAML files in az-cli (including root folder) with JSON files in az-cli-json."""
    yaml_files = get_yaml_json_files(yaml_root, (".yml", ".yaml"))
    json_files = get_yaml_json_files(json_root, (".json",))

    missing_json_files = []

    # ✅ Step 1: Ensure the script checks files in az-cli ROOT folder
    yaml_root_files = [f for f in os.listdir(yaml_root) if f.endswith(('.yml', '.yaml'))]
    json_root_files = [f for f in os.listdir(json_root) if f.endswith('.json')]

    # ✅ Step 2: Compare files in the ROOT folder of az-cli
    for yaml_file in yaml_root_files:
        json_file = yaml_file.replace(".yaml", ".json").replace(".yml", ".json")
        json_path = os.path.join(json_root, json_file)

        print(f"🔍 Checking YAML: {os.path.join(yaml_root, yaml_file)}")  # Debug print
        print(f"🔍 Expected JSON: {json_path}")  # Debug print

        if json_file not in json_root_files:
            missing_json_files.append(f"Missing JSON File: {json_path}")
            logging.warning(f"Missing JSON File: {json_path}")

    # ✅ Step 3: Compare YAML & JSON files inside subfolders
    for folder, yaml_list in yaml_files.items():
        json_folder = os.path.join(json_root, folder)

        for yaml_file in yaml_list:
            json_file = yaml_file.replace(".yaml", ".json").replace(".yml", ".json")
            json_path = os.path.join(json_folder, json_file)

            print(f"🔍 Checking YAML: {os.path.join(yaml_root, folder, yaml_file)}")  # Debug print
            print(f"🔍 Expected JSON: {json_path}")  # Debug print

            if folder in json_files:
                if json_file not in json_files[folder]:
                    missing_json_files.append(f"Missing JSON File: {json_path}")
                    logging.warning(f"Missing JSON File: {json_path}")
            else:
                missing_json_files.append(f"Missing JSON File: {json_path}")
                logging.warning(f"Missing JSON File: {json_path}")

    # ✅ Print Summary
    print(f"\n✅ Total YAML Files Scanned (Including Root Folder): {sum(len(files) for files in yaml_files.values()) + len(yaml_root_files)}")
    print(f"✅ Total JSON Files Scanned (Including Root Folder): {sum(len(files) for files in json_files.values()) + len(json_root_files)}")

    if missing_json_files:
        print(f"\n❌ Mismatch Found: {len(missing_json_files)} JSON files missing!")
        print("Check `yaml_json_comparison_log.log` for details.")
    else:
        print("\n✅ All YAML files have corresponding JSON files!")

# Setup logging
setup_logging()

# Run Folder Comparison
compare_folders(yaml_root, json_root)

# Run YAML-JSON File Comparison
compare_yaml_json_files(yaml_root, json_root)
