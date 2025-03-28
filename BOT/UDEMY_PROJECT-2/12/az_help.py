import os
import subprocess
import logging
import sys

# Constants for logging and configuration
LOG_DIRECTORY = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\12\logs'
LOG_FILENAME = 'az_help_processing.log'
OUTPUT_FILENAME = 'az_help_output.txt'  # File to save the formatted output

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

# Run a shell command and return the output
def run_command(command):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running command: {e}")
        terminate_script(1, f"Failed to run command: {command}")

# Format the raw output from az --help into structured sections
def format_help_output(raw_output):
    """Format the raw output from az --help into structured sections."""
    lines = raw_output.splitlines()
    formatted_output = {
        'command': '',
        'subgroups': [],
        'commands': [],
        'arguments': [],
        'global_arguments': [],
        'examples': []
    }
    
    current_section = None
    last_argument = None  # Track the last argument for multi-line descriptions
    last_example = None   # Track the last example for multi-line continuation

    for line in lines:
        line = line.strip()
        if line.startswith('Group'):
            # Capture only the group description (skip the word "Group")
            group_index = lines.index(line)
            if group_index + 1 < len(lines):
                formatted_output['command'] = lines[group_index + 1]
            current_section = 'command'
        elif line.startswith('Subgroups'):
            current_section = 'subgroups'
        elif line.startswith('Commands'):
            current_section = 'commands'
        elif line.startswith('Arguments'):
            current_section = 'arguments'
        elif line.startswith('Global Arguments'):
            current_section = 'global_arguments'
        elif line.startswith('Examples'):
            current_section = 'examples'
        elif line:  # Non-empty lines
            if current_section == 'subgroups':
                formatted_output['subgroups'].append(f"  {line}")
            elif current_section == 'commands':
                formatted_output['commands'].append(f"  {line}")
            elif current_section == 'arguments' or current_section == 'global_arguments':
                if line.startswith("--"):  # New argument starts with --
                    formatted_output[current_section].append(f"  {line}")
                    last_argument = formatted_output[current_section][-1]  # Track the last argument
                else:
                    # Handle multi-line argument descriptions (continuation)
                    if last_argument:
                        last_argument += f"\n    {line}"  # Append the continuation with proper indentation
            elif current_section == 'examples':
                if line.startswith("az"):
                    formatted_output['examples'].append(f"{line}\n")  # New example with extra line breaks
                    last_example = formatted_output['examples'][-1]  # Track the last example for continuation
                else:
                    if last_example:
                        last_example += f" {line}"  # Append continuation lines for examples

    return formatted_output

# Save the formatted output to a file
def save_output_to_file(formatted_output, filename):
    """Save the formatted output to a file in UTF-8 format."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\nCommand:\n")
        f.write(f"{formatted_output['command']}\n")
        
        # Only write Subgroups if present
        if formatted_output['subgroups']:
            f.write("\nSubgroups:\n")
            for subgroup in formatted_output['subgroups']:
                f.write(f"{subgroup}\n")
        
        # Only write Commands if present
        if formatted_output['commands']:
            f.write("\nCommands:\n")
            for command in formatted_output['commands']:
                f.write(f"{command}\n")
        
        # Only write Arguments if present
        if formatted_output['arguments']:
            f.write("\nArguments:\n")
            for arg in formatted_output['arguments']:
                f.write(f"{arg}\n")
        
        # Only write Global Arguments if present
        if formatted_output['global_arguments']:
            f.write("\nGlobal Arguments:\n")
            for global_arg in formatted_output['global_arguments']:
                f.write(f"{global_arg}\n")
        
        # Only write Examples if present, with enhanced formatting
        if formatted_output['examples']:
            f.write("\nExamples:\n")
            for example in formatted_output['examples']:
                formatted_example = example.strip().replace("\\", "\n    ")  # Add indentation for multi-line examples
                f.write(f"{formatted_example}\n\n")  # Double new line between examples for readability

# Main function to run the script for one command
def main():
    setup_logging()

    # Example CLI command to run with --help
    command = input("Enter the Azure CLI command you want to run with --help: ")
    
    # Run the --help command and capture the output
    raw_output = run_command(f"{command} --help")
    
    if raw_output:
        # Format the raw output
        formatted_output = format_help_output(raw_output)
        
        # Save the formatted output to a file
        output_file = os.path.join(LOG_DIRECTORY, OUTPUT_FILENAME)
        save_output_to_file(formatted_output, output_file)
        print(f"Formatted output saved to {output_file}")

if __name__ == "__main__":
    main()
