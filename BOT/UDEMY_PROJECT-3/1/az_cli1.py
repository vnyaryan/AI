import os
import subprocess
import logging
import sys
import time

# Constants for logging and configuration
LOG_DIRECTORY = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-3\1'
LOG_FILENAME = 'az_help_processing.log'
OUTPUT_FILENAME = 'az_help_output.txt'  # File to save the raw output
INPUT_FILE = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-3\1\az_cli_list.txt'  # Input file containing list of Azure CLI commands
COMMAND_TIMEOUT = 60  # Timeout in seconds for running the az --help command

# Setup logging
def setup_logging():
    """Set up logging for the script."""
    try:
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(LOG_DIRECTORY, LOG_FILENAME),
                            filemode='a')  # Append mode
        logging.info("Logging setup successful.")
    except Exception as e:
        terminate_script(1, "Error setting up logging.")

# Terminate the script and log an error message if provided
def terminate_script(exit_code=1, message=None):
    """Terminate the script and log the error message if provided."""
    if message:
        logging.error(message)
    sys.exit(exit_code)

# Run a shell command with a timeout and return the output
def run_command(command):
    """Run a shell command and return the output with a timeout."""
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=COMMAND_TIMEOUT)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        logging.error(f"Command timed out: {command}. Waiting for 60 seconds before continuing.")
        time.sleep(60)  # Wait for 60 seconds before continuing
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running command: {e}. Waiting for 60 seconds before continuing.")
        time.sleep(60)  # Wait for 60 seconds before continuing

# Save the raw output to a file
def save_output_to_file(raw_output, filename):
    """Save the raw output to a file in UTF-8 format."""
    with open(filename, 'a', encoding='utf-8') as f:  # Append mode
        f.write(raw_output + "\n")
        f.write("\n" + "=" * 120 + "\n" + "=" * 120 + "\n\n")  # Add separators for readability

# Process the input file containing the list of Azure CLI commands
def process_input_file(input_file):
    """Process each command from the input file and run az --help."""
    if not os.path.exists(input_file):
        logging.error(f"Input file {input_file} does not exist.")
        terminate_script(1, "Input file not found.")
    
    with open(input_file, 'r') as file:
        commands = [line.strip() for line in file.readlines() if line.strip()]
    
    if not commands:
        logging.error("Input file is empty or contains no valid commands.")
        terminate_script(1, "No commands found in input file.")
    
    return commands

# Main function to run the script for multiple commands from input file
def main():
    setup_logging()

    # Process the input file to get the list of Azure CLI commands
    commands = process_input_file(INPUT_FILE)

    # Process each command and run --help
    for command in commands:
        logging.info(f"Processing command: {command}")
        
        # Run the --help command and capture the output
        raw_output = run_command(f"{command} --help")
        
        if raw_output:
            # Save the raw output to a file
            output_file = os.path.join(LOG_DIRECTORY, OUTPUT_FILENAME)
            save_output_to_file(raw_output, output_file)
            logging.info(f"Raw output for {command} saved to {output_file}")

if __name__ == "__main__":
    main()
