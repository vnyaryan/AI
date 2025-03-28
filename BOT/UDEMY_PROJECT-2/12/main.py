import os
import logging
import sys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from az_help import run_command, format_help_output, save_output_to_file
import json

# Constants for logging and OpenAI configuration
LOG_DIRECTORY = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\12\logs'
LOG_FILENAME = 'phase1.log'
OUTPUT_FILENAME = 'az_help_output.txt'
OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Load the list of valid Azure CLI commands from a text file
def load_command_list():
    with open('az_monitor_list.txt', 'r') as file:
        return [line.strip() for line in file.readlines()]

# Set up logging to track script execution
def setup_logging():
    try:
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(LOG_DIRECTORY, LOG_FILENAME),
                            filemode='a')
        logging.info("Logging setup successful.")
    except Exception as e:
        terminate_script(1, "Error setting up logging.")

# Terminate the script and log the error message if provided
def terminate_script(exit_code=1, message=None):
    if message:
        logging.error(message)
    sys.exit(exit_code)

# Interact with OpenAI and retrieve the CLI command and parameters using LangChain
def get_command_and_parameters(query):
    try:
        system_message_template = """You are an expert in Azure CLI. Your task is to identify the Azure CLI command 
        and its required parameters based on the user's query. Return the result as a JSON object with 
        'command' and 'parameters' fields."""
        human_message_template = "{task}"

        system_message_prompt_template = SystemMessagePromptTemplate.from_template(system_message_template)
        human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_message_template)

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message_prompt_template, human_message_prompt_template]
        )

        formatted_prompt = chat_prompt_template.format_prompt(task=query)
        final_prompt_messages = formatted_prompt.to_messages()

        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        llm_response = llm.invoke(final_prompt_messages)

        return llm_response.content.strip()
    except Exception as e:
        logging.error(f"Error while querying OpenAI: {e}")
        terminate_script(1, "Failed to query OpenAI.")

# Validate the command with the predefined list of valid commands
def validate_command(command, valid_commands):
    return command in valid_commands

# Present the identified command and parameters to the user
def present_command_details(command_details):
    try:
        # Assuming the response is returned as a JSON string from OpenAI
        command_data = json.loads(command_details)

        print("\nHere is the Azure CLI command and its parameters based on your query:")
        print(f"Command: {command_data['command']}")
        print("Parameters:")

        # Correctly handle the list of parameters
        if isinstance(command_data['parameters'], list):
            for param in command_data['parameters']:
                print(f"  {param}")
        else:
            print("  No valid parameters returned.")
    except Exception as e:
        logging.error(f"Error while presenting command details: {e}")
        terminate_script(1, "Failed to present command details.")

# Main chatbot logic for Phase-1
def main():
    setup_logging()

    user_query = input("Please describe the Azure task you'd like to perform: ")

    # Load the valid command list
    valid_commands = load_command_list()

    # Get the actual response from OpenAI
    command_response = get_command_and_parameters(user_query)

    # Present the actual extracted command and parameters
    command_data = json.loads(command_response)
    present_command_details(command_response)

    # Validate the command identified by OpenAI
    if validate_command(command_data['command'], valid_commands):
        print(f"The command '{command_data['command']}' is valid. Fetching detailed information...")
        
        # Run the az --help for the identified command using az_help.py
        raw_output = run_command(f"{command_data['command']} --help")

        if raw_output:
            formatted_output = format_help_output(raw_output)
            output_file = os.path.join(LOG_DIRECTORY, OUTPUT_FILENAME)
            save_output_to_file(formatted_output, output_file)
            print(f"Formatted output saved to {output_file}")
    else:
        print(f"The command '{command_data['command']}' is not valid. Please check your input.")

if __name__ == "__main__":
    main()
