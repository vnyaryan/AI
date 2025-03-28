def find_lines_with_equals(filename, output_file):
    target_pattern = '=' * 120  # 120 equal signs (can be adjusted based on your requirement)

    try:
        # Open the input file and read line by line
        with open(filename, 'r', encoding='utf-8') as file:
            # Open the output file in write mode
            with open(output_file, 'w') as outfile:
                for line_number, line in enumerate(file, start=1):
                    # Check if the line contains exactly 120 equal signs
                    if line.strip() == target_pattern:
                        outfile.write(f"{line_number}\n")  # Write the line number to the output file
        print(f"Line numbers written to {output_file}")
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    input_file = 'az_help_output-5.txt'
    output_file = 'line_details.txt'
    find_lines_with_equals(input_file, output_file)
