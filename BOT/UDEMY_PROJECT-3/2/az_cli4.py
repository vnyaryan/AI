import os

# Define the directory where the new files will be saved
LOG_DIRECTORY = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-3\2\chunk_details'

# Ensure the directory exists
os.makedirs(LOG_DIRECTORY, exist_ok=True)

def get_lines_between(filename, start_line, end_line, output_file):
    """Reads content from the specified line range and writes it to a new file."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            with open(output_file, 'w', encoding='utf-8') as final_output:  # 'w' for write mode (new file each time)
                for line_number, line in enumerate(file, start=1):
                    # Write the lines that fall within the specified range
                    if start_line <= line_number <= end_line:
                        final_output.write(line)
        print(f"Content from line {start_line} to line {end_line} written to {output_file}")
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def process_line_details(line_details_file, source_file):
    """Reads line numbers from line_details.txt and writes content from the source file to new files."""
    try:
        # Read the line numbers from line_details.txt
        with open(line_details_file, 'r', encoding='utf-8') as file:
            lines = [int(line.strip()) for line in file.readlines()]

        # Iterate through the line numbers in pairs (1st and 2nd, 2nd and 3rd, etc.)
        for i in range(len(lines) - 1):
            start_line = lines[i] + 1    # Starting line (next line after the current line number)
            end_line = lines[i + 1] - 1  # Ending line (just before the next line number)

            # Generate a unique filename for each chunk
            output_file = os.path.join(LOG_DIRECTORY, f'chunk_{start_line}_to_{end_line}.txt')

            get_lines_between(source_file, start_line, end_line, output_file)

    except FileNotFoundError:
        print(f"Error: The file '{line_details_file}' does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    line_details_file = 'line_details.txt'  # File with line numbers
    source_file = 'az_help_output-5.txt'    # Source file to extract content
    process_line_details(line_details_file, source_file)
