# extract_commands_and_arguments.py

import re
import sys

def extract_commands(lines, keyword="Command"):
    """
    Extracts command names from the 'Command' section.
    
    Parameters:
        lines (list): List of lines from the input file.
        keyword (str): The keyword to identify the Command section.
    
    Returns:
        list: A list of extracted command names.
    """
    commands = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == keyword:
            i += 1  # Move to the next line after 'Command'
            # Continue reading lines until a non-indented line or end of file
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip() == "":
                    i += 1
                    continue  # Skip empty lines
                if not next_line.startswith(" ") and not next_line.startswith("\t"):
                    break  # Non-indented line signifies end of command section
                # Assuming the command line has the format: command : description
                parts = next_line.strip().split(":", 1)
                if len(parts) >= 1:
                    command = parts[0].strip()
                    if command:  # Ensure it's not empty
                        commands.append(command)
                i += 1
        else:
            i += 1  # Move to the next line if 'Command' not found
    return commands

def extract_arguments(lines, start_keyword="Arguments", stop_keyword="Global Arguments"):
    """
    Extracts the first argument starting with '--' from each line in the 'Arguments' section.
    
    Parameters:
        lines (list): List of lines from the input file.
        start_keyword (str): The keyword to start extraction.
        stop_keyword (str): The keyword to stop extraction.
    
    Returns:
        list: A list of extracted arguments.
    """
    arguments = []
    in_target_section = False  # Flag to indicate if we're within the target section

    for line in lines:
        stripped_line = line.strip()

        # Check if the current line is the start keyword
        if stripped_line == start_keyword:
            in_target_section = True
            continue  # Move to the next line after the header

        # Check if we've reached the stop keyword
        if stripped_line == stop_keyword:
            in_target_section = False
            break  # Exit the loop as we've reached the stopping point

        # If we're within the target section, process the lines
        if in_target_section:
            # Skip empty lines
            if stripped_line == "":
                continue

            # Check if the line starts with '--'
            if stripped_line.startswith("--"):
                # Use regex to find the first argument starting with '--'
                found_arg = re.search(r'(--[\w-]+)', stripped_line)
                if found_arg:
                    arguments.append(found_arg.group(1))

    return arguments

def write_output(output_file, commands, arguments):
    """
    Writes the extracted commands and arguments to the output file with headers.
    
    Parameters:
        output_file (str): Path to the output file.
        commands (list): List of extracted commands.
        arguments (list): List of extracted arguments.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            if commands:
                outfile.write("Commands:\n")
                for cmd in commands:
                    outfile.write(cmd + '\n')
                outfile.write("\n")  # Add a newline for separation
            else:
                outfile.write("Commands:\nNo commands found.\n\n")
            
            if arguments:
                outfile.write("Arguments:\n")
                for arg in arguments:
                    outfile.write(arg + '\n')
            else:
                outfile.write("Arguments:\nNo arguments found.\n")
        
        print(f"Successfully wrote extracted data to '{output_file}'.")
    except Exception as e:
        print(f"Error writing to '{output_file}': {e}")

def main():
    input_filename = "input.txt"
    output_filename = "output.txt"

    try:
        with open(input_filename, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' does not exist.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading '{input_filename}': {e}")
        sys.exit(1)

    # Extract Commands
    commands = extract_commands(lines)
    
    # Extract Arguments
    arguments = extract_arguments(lines)
    
    # Remove duplicates while preserving order
    unique_commands = []
    seen_commands = set()
    for cmd in commands:
        if cmd not in seen_commands:
            seen_commands.add(cmd)
            unique_commands.append(cmd)
    
    unique_arguments = []
    seen_arguments = set()
    for arg in arguments:
        if arg not in seen_arguments:
            seen_arguments.add(arg)
            unique_arguments.append(arg)
    
    # Write to output.txt
    write_output(output_filename, unique_commands, unique_arguments)

if __name__ == "__main__":
    main()
