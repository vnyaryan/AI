import os

def remove_ai_knowledge_base_lines(input_file, output_file):
    try:
        # Check if the input file exists
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist.")
            return

        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                # If the line starts with the unwanted phrase, skip it
                if not line.startswith("To search AI knowledge base for examples, use:"):
                    outfile.write(line)

        print(f"Processing completed. Cleaned content saved to {output_file}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Provide the input and output file paths
input_file = 'az_help_output-2.txt'
output_file = 'az_help_output-3.txt'

remove_ai_knowledge_base_lines(input_file, output_file)
