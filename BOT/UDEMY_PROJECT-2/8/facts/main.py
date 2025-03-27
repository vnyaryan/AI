import requests
from bs4 import BeautifulSoup
import logging
import os

# Constants for logging and output
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\8\facts\logs'
log_filename = 'webpage_extraction.log'
output_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\8\facts'
output_filename = 'command_summary.txt'

# URL of the Azure CLI alias page
url = "https://learn.microsoft.com/en-us/cli/azure/account/alias?view=azure-cli-latest"

# Function to set up logging
def setup_logging():
    try:
        os.makedirs(log_directory, exist_ok=True)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='w')
        logging.debug("Logging setup completed.")
    except Exception as e:
        logging.error(f"Error setting up logging: {e}")

# Function to fetch the webpage content
def fetch_webpage(url):
    try:
        logging.info(f"Fetching webpage: {url}")
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        logging.info("Successfully retrieved webpage content.")
        return response.text
    except Exception as e:
        logging.error(f"Failed to retrieve webpage: {e}")
        return None

# Function to parse and extract command information
def extract_command_info(soup):
    commands_info = []

    # Locate sections by using BeautifulSoup's capabilities (modify selectors based on the actual webpage structure)
    command_sections = soup.find_all('h2')  # Modify this based on the actual structure of the webpage
    for command_section in command_sections:
        command_info = {}
        
        # Extracting the command name
        command_info['name'] = command_section.get_text().strip()

        # Extract the brief description (assuming it follows the command name)
        description_section = command_section.find_next('p')
        if description_section:
            command_info['description'] = description_section.get_text().strip()
        else:
            command_info['description'] = "No description available"

        # Extract global parameters, required parameters, optional parameters, and examples
        command_info['global_parameters'] = extract_parameters(command_section, "Global Parameters")
        command_info['required_parameters'] = extract_parameters(command_section, "Required Parameters")
        command_info['optional_parameters'] = extract_parameters(command_section, "Optional Parameters")
        command_info['examples'] = extract_examples(command_section)

        commands_info.append(command_info)

    return commands_info

# Helper function to extract parameters
def extract_parameters(command_section, parameter_type):
    parameters = []
    
    # Locate the parameter table or list
    parameter_section = command_section.find_next(['h3', 'h4'], string=parameter_type)  # This looks for headings (h3 or h4)
    
    if parameter_section:
        param_table = parameter_section.find_next('table')
        if param_table:
            rows = param_table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    param_name = cols[0].get_text().strip()
                    param_description = cols[1].get_text().strip()
                    parameters.append(f"{param_name}: {param_description}")
    
    return parameters

# Helper function to extract examples
def extract_examples(command_section):
    examples = []
    example_section = command_section.find_next('h3', string="Examples")
    if example_section:
        example_list = example_section.find_next('pre')
        if example_list:
            examples.append(example_list.get_text().strip())
    return examples

# Function to save the summarized information to a file
def save_summary_to_file(commands_info):
    try:
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for command in commands_info:
                logging.info(f"Processing command: {command['name']}")
                f.write(f"Command: {command['name']}\n")
                f.write(f"Description: {command['description']}\n\n")
                
                f.write("Global Parameters:\n")
                if command['global_parameters']:
                    for param in command['global_parameters']:
                        f.write(f"  - {param}\n")
                else:
                    f.write("  None\n")
                
                f.write("\nRequired Parameters:\n")
                if command['required_parameters']:
                    for param in command['required_parameters']:
                        f.write(f"  - {param}\n")
                else:
                    f.write("  None\n")
                
                f.write("\nOptional Parameters:\n")
                if command['optional_parameters']:
                    for param in command['optional_parameters']:
                        f.write(f"  - {param}\n")
                else:
                    f.write("  None\n")
                
                f.write("\nExamples:\n")
                if command['examples']:
                    for example in command['examples']:
                        f.write(f"  {example}\n")
                else:
                    f.write("  None\n")
                
                f.write("\n" + "-"*50 + "\n")
                
        logging.info(f"Summary saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving summary to file: {e}")

def main():
    setup_logging()
    
    webpage_content = fetch_webpage(url)
    
    if webpage_content:
        soup = BeautifulSoup(webpage_content, 'html.parser')
        commands_info = extract_command_info(soup)
        save_summary_to_file(commands_info)
    else:
        logging.error("Failed to retrieve webpage content.")

if __name__ == "__main__":
    main()
