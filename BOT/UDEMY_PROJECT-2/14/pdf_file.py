import fitz  # PyMuPDF

def is_valid_azure_command(line):
    """
    Helper function to check if a line is a valid Azure CLI command.
    Valid commands start with 'az', do not contain unwanted strings, and are not parameters.
    """
    # The line should start with 'az', have proper formatting, and should not include unwanted terms
    if line.startswith("az ") and "|" not in line and "Microsoft Learn" not in line:
        # Ignore lines containing parameters or options indicated by --, -, [], "", (), <> and specific parameter flags (-u, -p, -g, -n)
        if "--" not in line and "-" not in line.split()[1:] and "[" not in line and "]" not in line \
                and '"' not in line and '(' not in line and ')' not in line and '<' not in line and '>' not in line \
                and "-u" not in line and "-p" not in line and "-g" not in line and "-n" not in line:
            return True
    return False


def extract_azure_cli_commands(pdf_path, output_file):
    """
    Extracts unique Azure CLI command names (lines starting with 'az') from the PDF.
    Filters out duplicates, parameters, and non-command lines. Saves the commands to a text file.
    """
    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)

        commands_set = set()  # Use a set to keep only unique commands

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Extracted Unique Azure CLI Commands:\n")
            f.write("-" * 40 + "\n")

            # Iterate over each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")

                # Split the page text into lines
                lines = text.split('\n')

                # Check each line and capture valid Azure CLI commands
                for line in lines:
                    line = line.strip()  # Remove leading and trailing spaces

                    if is_valid_azure_command(line):
                        commands_set.add(line)  # Add the command to the set (ignores duplicates)

            # Write the unique commands to the file
            for command in sorted(commands_set):  # Sort the commands alphabetically for easier reading
                f.write(f"{command}\n")

        print(f"Unique Azure CLI commands extracted and saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Specify the path to your PDF file and the output file
pdf_path = r"C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\14\az_module.pdf"
output_file = r"C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\14\extracted_unique_azure_cli_commands.txt"

# Run the extraction function
extract_azure_cli_commands(pdf_path, output_file)


