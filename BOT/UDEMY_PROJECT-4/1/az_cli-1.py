"""
This script interacts with the OpenAI API to generate, refine, and validate Azure CLI commands based on user input.

Logic of the script:
1. Sets up logging configuration for tracking script execution and errors.
2. Generates an initial Azure CLI command based on the user's task.
3. Prompts the user for feedback to determine if the generated command is correct.
4. If the user provides feedback indicating the command needs changes, the script refines the command based on user clarification.
5. Repeats the feedback loop until the user approves the command.
6. Logs the success or failure of each operation and saves the conversation history.

Input Parameters:
1. API_KEY: The OpenAI API key for authentication (set as a constant at the start of the script).
2. INITIAL_TASK: A string describing the initial Azure CLI task (e.g., "list all resource groups in a subscription").

Output:
1. Displays the generated Azure CLI command to the user.
2. Refines and displays updated versions of the command based on user feedback.
3. Outputs the final approved command once the user confirms correctness.
4. Logs details of the interaction and stores the conversation history for further review.
"""

import os
import logging
import sys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationSummaryMemory

# Constants for logging and API setup
LOG_DIRECTORY = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-3\3'
LOG_FILENAME = 'az_cli-1.log'
API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"  # Replace with your actual API key.
INITIAL_TASK = "list all resource groups in a subscription"

# Set up logging configuration to track script execution
def setup_logging():
    try:
        # Ensure the log directory exists
        os.makedirs(LOG_DIRECTORY, exist_ok=True)

        # Configure logging settings
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(LOG_DIRECTORY, LOG_FILENAME),
            filemode='a'  # Append mode
        )
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        sys.exit(1)  # Ensure that the script exits if logging setup fails.


# Terminate the script with an optional error message
def terminate_script(exit_code=1, message=None):
    if message:
        logging.error(message)
    sys.exit(exit_code)

# Generate an initial Azure CLI command based on the user task
def generate_azure_cli_command(task, llm, memory):
    logging.debug("Generating Azure CLI command for task: %s", task)
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are an expert of Azure and az cli. Your task is to provide the appropriate Azure az cli command as per the details provided by the user."
            ),
            HumanMessagePromptTemplate.from_template("{task}")
        ]
    )
    azure_cli_request_prompt = chat_prompt_template.format_prompt(task=task).to_messages()
    history_messages = memory.load_memory_variables({'input': 'messages'})['messages']
    azure_cli_request_prompt = history_messages + azure_cli_request_prompt
    response = llm.invoke(azure_cli_request_prompt)
    memory.save_context({'input': azure_cli_request_prompt}, {'output': response.content})
    return response.content

# Get feedback from the user to refine the command
def get_user_feedback():
    return input("\nIs the generated Azure CLI command correct? (yes/no): ").strip().lower()

# Refine the Azure CLI command based on user feedback
def refine_azure_cli_command(clarification, llm, memory):
    logging.debug("Refining Azure CLI command with clarification: %s", clarification)
    clarification_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are helping refine an Azure CLI command."
            ),
            HumanMessagePromptTemplate.from_template(
                "The user suggested the following changes to the command: {clarification}"
            )
        ]
    )
    clarification_prompt = clarification_prompt_template.format_prompt(clarification=clarification).to_messages()
    history_messages = memory.load_memory_variables({'input': 'messages'})['messages']
    refinement_request_prompt = history_messages + clarification_prompt
    response = llm.invoke(refinement_request_prompt)
    memory.save_context({'input': refinement_request_prompt}, {'output': response.content})
    return response.content

# Main function to drive the interaction with the user
def main():
    setup_logging()

    # Initialize the LLM and memory
    llm = ChatOpenAI(openai_api_key=API_KEY)
    memory = ConversationSummaryMemory(
        memory_key="messages",
        return_messages=True,
        llm=llm
    )
    logging.debug("ChatOpenAI initialized with provided API key.")

    # Generate the initial Azure CLI command
    azure_cli_command = generate_azure_cli_command(INITIAL_TASK, llm, memory)
    print("Generated Azure CLI Command:")
    print(azure_cli_command)

    # Get user feedback and refine the command if needed
    user_feedback = get_user_feedback()
    while user_feedback != "yes":
        clarification = input("What would you like to change or clarify in the command?: ").strip()
        azure_cli_command = refine_azure_cli_command(clarification, llm, memory)
        print("\nRefined Azure CLI Command:")
        print(azure_cli_command)
        user_feedback = get_user_feedback()

    # Output the final command after user approval
    print("\nFinal Azure CLI Command:")
    print(azure_cli_command)
    logging.debug("User approved the Azure CLI command.")

    # Retrieve and display the conversation summary memory content
    summary = memory.load_memory_variables({'input': 'messages'})['messages']
    if summary:
        print("\nConversation Summary Memory Content:")
        for msg in summary:
            print(f"{msg.type.capitalize()}: {msg.content}")
    else:
        print("No summary available or conversation did not produce a summary.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("An error occurred: %s", e)
        print(f"An error occurred: {e}")
