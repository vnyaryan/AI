# Define the input and output file paths
input_file = 'az_help_output-3.txt'  # Replace with your actual input file path
output_file = 'az_help_output-4.txt'  # Replace with the desired output file path

# Open the input file for reading and the output file for writing
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    in_examples_section = False  # Flag to track if we are in the "Examples" section

    for line in infile:
        # Detect the separator line that marks the end of a section
        if line.startswith("========================================================================================================================"):
            in_examples_section = False  # Stop skipping lines when the separator is found
            outfile.write(line)  # Write the separator line to the output file
            continue

        # If the line contains exactly "Examples" (ignoring leading/trailing spaces), set the flag to True
        if line.strip() == "Examples":
            in_examples_section = True  # Start skipping the "Examples" section and following lines
            continue  # Skip the "Examples" heading itself

        # Write the line if we are not in the "Examples" section
        if not in_examples_section:
            outfile.write(line)

print(f"'Examples' sections and all content until the next separator have been removed.")
