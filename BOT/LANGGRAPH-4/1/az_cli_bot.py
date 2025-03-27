import json
import os
import sys
import logging
from openai import OpenAI  # Correct way to import the OpenAI client

# ----------------------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------------------
LOG_DIRECTORY = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\LANGGRAPH-4\1\log'
LOG_FILENAME = "az_cli_bot.log"
INPUT_FILE = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\LANGGRAPH-4\1\az-cli-json\account\subscription.json'
OUTPUT_FILE = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\LANGGRAPH-4\1\qa_output.txt'
API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

def setup_logging():
    try:
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(LOG_DIRECTORY, LOG_FILENAME),
            filemode='w'
        )
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        sys.exit(1)

setup_logging()

# Initialize OpenAI client properly for version 1.0.0+
client = OpenAI(api_key=API_KEY)

# Define the 10 Q&A categories along with their category-specific instructions
qa_categories = {
    "Overview Questions": "Generate an overview question and answer that explains the purpose and functionality of this command.",
    "Syntax Questions": "Generate a question and answer that details the correct syntax and usage format of this command.",
    "Parameter Questions": "Generate a question and answer that explains the required and optional parameters of this command.",
    "Usage Example Questions": "Generate a question and answer that provides a practical usage example for this command.",
    "Best Practices/Tips Questions": "Generate a question and answer that offers best practices or tips when using this command.",
    "General Overview Questions": "Generate a question and answer that gives a general overview of Azure CLI subscriptions, especially focusing on this command group.",
    "Clarification/Follow-up Questions": "Generate a question and answer that prompts for clarification or guides the user if the command usage is ambiguous.",
    "Contextual Help Questions": "Generate a question and answer that provides contextual help for managing subscriptions using this command.",
    "FAQ-Style Q&A": "Generate a frequently asked question and answer that addresses common issues or misconceptions about this command.",
    "Fallback Answers": "Generate a fallback question and answer that offers multiple options or asks the user to choose from related commands if the query is unclear."
}

def generate_qa(command_details, category, instruction):
    """
    Generate a Q&A pair using the OpenAI Chat API.
    
    Args:
        command_details (dict): Dictionary containing the command details.
        category (str): The Q&A category type.
        instruction (str): Category-specific instruction.
        
    Returns:
        str: The generated Q&A pair as text, or None if an error occurs.
    """
    # Build instructions and input text for the API call.
    instructions_text = (
        f"Generate a Q&A pair for the following Azure CLI command details for the category '{category}'. "
        f"Follow these instructions: {instruction} "
        "Provide the output in the format:\nQ: <question>\nA: <answer>"
    )
    
    input_text = (
        f"Command Name: {command_details.get('name', 'N/A')}\n"
        f"Summary: {command_details.get('summary', 'N/A')}\n"
        f"Syntax: {command_details.get('syntax', 'N/A')}\n"
        f"Required Parameters: {', '.join(command_details.get('required_parameters', [])) or 'None'}\n"
        f"Optional Parameters: {', '.join(command_details.get('optional_parameters', [])) or 'None'}\n"
        f"Examples: {', '.join(command_details.get('examples', [])) or 'None'}\n"
    )
    
    try:
        logging.info(f"Sending API request for command '{command_details.get('name', 'N/A')}', category '{category}'.")
        
        # Correct API call for OpenAI >=1.0.0
        response = client.chat.completions.create(
            model="gpt-4o",  # adjust the model if necessary
            messages=[
                {"role": "system", "content": instructions_text},
                {"role": "user", "content": input_text}
            ],
            temperature=0.7,
            max_tokens=150
        )

        output_text = response.choices[0].message.content.strip()
        logging.info(f"Received Q&A for command '{command_details.get('name', 'N/A')}', category '{category}'.")
        return output_text

    except Exception as e:
        logging.error(f"Error generating Q&A for command '{command_details.get('name', 'N/A')}', category '{category}': {str(e)}")
        return None

def main():
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
        logging.info(f"Loaded JSON file '{INPUT_FILE}' successfully.")
    except Exception as e:
        logging.error(f"Error loading JSON file '{INPUT_FILE}': {str(e)}")
        sys.exit(1)
    
    commands = data.get("commands", [])
    if not commands:
        logging.warning("No commands found in the JSON file.")
    
    all_output = []
    
    # Process each command and each Q&A category with error handling.
    for cmd in commands:
        command_header = f"Command: {cmd.get('name', 'N/A')}\n{'=' * 60}\n"
        all_output.append(command_header)
        for category, instruction in qa_categories.items():
            try:
                qa_pair = generate_qa(cmd, category, instruction)
                if qa_pair:
                    entry = f"Category: {category}\n{qa_pair}\n{'-' * 40}\n"
                    all_output.append(entry)
                else:
                    logging.warning(f"No Q&A generated for command '{cmd.get('name', 'N/A')}', category '{category}'.")
            except Exception as e:
                logging.error(f"Exception processing command '{cmd.get('name', 'N/A')}', category '{category}': {str(e)}")
    
    try:
        with open(OUTPUT_FILE, 'w') as f:
            f.write("\n".join(all_output))
        logging.info(f"Q&A pairs saved to '{OUTPUT_FILE}'.")
    except Exception as e:
        logging.error(f"Error writing to output file '{OUTPUT_FILE}': {str(e)}")

if __name__ == "__main__":
    main()
