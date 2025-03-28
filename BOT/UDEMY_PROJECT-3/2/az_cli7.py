import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ------------------------
# Variables
# ------------------------
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-3\2'
log_filename = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-3\2\log\az_cli7.log'
api_key = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"
folder_path = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-3\2\chunk_details'
summary_folder_path = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-3\2\az_cli_summary'
names_file = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-3\2\name.txt'  


# ------------------------
# Logging Setup
# ------------------------
def setup_logging():
    try:
        os.makedirs(log_directory, exist_ok=True)
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename), filemode='a')
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to set up logging: {e}")

# ------------------------
# Initialize OpenAI 
# ------------------------
def initialize_llm(api_key):
    return ChatOpenAI(openai_api_key=api_key)

# ------------------------
# Define Templates and Summarization Function
# ------------------------
def create_chat_prompt():
    system_message_template = """You are an expert in Azure CLI. Your task is to provide a concise summary for each Azure CLI command or command group.
    - If the input describes a command group, provide an overview of the group's purpose.
    - If the input describes an individual command, provide a brief description of the command's purpose and utility.
    - Avoid listing subcommands, arguments, or global parameters.
    Return the summary as a short paragraph that a developer can quickly understand."""
    
    system_message_prompt_template = SystemMessagePromptTemplate.from_template(system_message_template)
    human_message_template = "{content}"
    human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_message_template)
    
    # Combine system and human message templates into a chat prompt template
    return ChatPromptTemplate.from_messages(
        [system_message_prompt_template, human_message_prompt_template]
    )

def summarize_command(llm, chat_prompt_template, file_content):
    # Format the prompt with the system and human messages
    final_prompt = chat_prompt_template.format_prompt(content=file_content).to_messages()
    
    # Send the message to LLM (invoke the LLM)
    llm_response = llm.invoke(final_prompt)
    
    return llm_response.content  # Return summarized response

# ------------------------
# Save LLM Response to File
# ------------------------
def save_summary_to_file(summary_folder_path, filename, summary_content):
    # Ensure the summary folder exists
    os.makedirs(summary_folder_path, exist_ok=True)
    
    # Construct the full path for the summary file
    summary_file_path = os.path.join(summary_folder_path, filename)
    
    # Save the summary content to a text file
    with open(summary_file_path, 'w') as summary_file:
        summary_file.write(summary_content)

# ------------------------
# Process Files based on file names from names_file
# ------------------------
def process_files_from_names_file(names_file, folder_path, summary_folder_path, llm, chat_prompt_template):
    try:
        with open(names_file, 'r') as f:
            file_names = [line.strip() for line in f.readlines()]  # Read all file names
        
        for filename in file_names:
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    file_content = file.read()  # Read the command details from the file
                    summary = summarize_command(llm, chat_prompt_template, file_content)  # Get summarized details
                    
                    logging.debug(f"Summarized Command for {filename}: {summary}")
                    #print(f"Summary for {filename}:\n{summary}\n")
                    
                    # Save the summary to a file with the same name as the CLI command file
                    save_summary_to_file(summary_folder_path, filename, summary)
            else:
                logging.warning(f"File not found: {file_path}")
    except Exception as e:
        logging.error(f"Error processing files: {e}")

# ------------------------
# Main Function
# ------------------------
def main():
    # Setup logging
    setup_logging()
    
    logging.debug("API key loaded successfully.")
    
    # Initialize OpenAI
    llm = initialize_llm(api_key)
    
    # Create chat prompt template
    chat_prompt_template = create_chat_prompt()
    
    logging.debug(f"Processing files listed in: {names_file}")
    
    # Process the files using the names from the names file and summarize each Azure CLI command
    process_files_from_names_file(names_file, folder_path, summary_folder_path, llm, chat_prompt_template)

# ------------------------
# Entry Point
# ------------------------
if __name__ == "__main__":
    main()
