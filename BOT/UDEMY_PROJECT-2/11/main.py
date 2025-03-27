import os
import logging
import sys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationSummaryMemory
import json

# Constants for logging and OpenAI configuration
LOG_DIRECTORY = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\11\logs'
LOG_FILENAME = 'phase1.log'
OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Set up logging to track script execution
def setup_logging():
    try:
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(LOG_DIRECTORY, LOG_FILENAME),
                            filemode='a')  # Append mode
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
        # Define system and human message templates
        system_message_template = """You are an expert in Azure CLI. Your task is to identify the Azure CLI command 
        and its required parameters based on the user's query. Return the result as a JSON object with 
        'command' and 'parameters' fields."""
        human_message_template = "{task}"

        # Convert the templates into structured prompt templates
        system_message_prompt_template = SystemMessagePromptTemplate.from_template(system_message_template)
        human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_message_template)

        # Combine system and human prompts into a unified ChatPromptTemplate
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message_prompt_template, human_message_prompt_template]
        )

        # Format the prompt by replacing the placeholders with the specific query/task from the user
        formatted_prompt = chat_prompt_template.format_prompt(task=query)

        # Convert the formatted prompt into a list of messages
        final_prompt_messages = formatted_prompt.to_messages()

        # Initialize the OpenAI model using LangChain's ChatOpenAI
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

        # Send the prompt to the LLM and get the response
        llm_response = llm.invoke(final_prompt_messages)

        return llm_response.content.strip()
    except Exception as e:
        logging.error(f"Error while querying OpenAI: {e}")
        terminate_script(1, "Failed to query OpenAI.")

# Present the identified command and parameters to the user
def present_command_details(command_details):
    try:
        # Assuming the response is returned as a JSON string from OpenAI
        # Convert the response into a Python dictionary
        command_data = json.loads(command_details)
        
        print("\nHere is the Azure CLI command and its parameters based on your query:")
        print(f"Command: {command_data['command']}")
        print("Parameters:")
        for param in command_data['parameters']:
            print(f"  {param}")
    except Exception as e:
        logging.error(f"Error while presenting command details: {e}")
        terminate_script(1, "Failed to present command details.")

# Main chatbot logic for Phase-1
def main():
    setup_logging()

    user_query = input("Please describe the Azure task you'd like to perform: ")

    # Get the actual response from OpenAI
    command_response = get_command_and_parameters(user_query)

    # Present the actual extracted command and parameters
    present_command_details(command_response)

if __name__ == "__main__":
    main()
