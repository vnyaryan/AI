import os
import logging
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_models import ChatPerplexity

# Step 1: Setup Logging
def setup_logging():
    """Setup logging configuration with a fixed log filename."""
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\2'
        log_filename = 'iac-agent-1.log'
        
        # Ensure the log directory exists
        os.makedirs(log_directory, exist_ok=True)

        # Configure logging
        logging.basicConfig(level=logging.DEBUG,  
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='w')
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")

setup_logging()

# Step 2: Define and structure message templates
system_message_template = "You are a helpful assistant that summarizes text documents."
human_message_template = "{input}"

system_prompt = SystemMessagePromptTemplate.from_template(system_message_template)
human_prompt = HumanMessagePromptTemplate.from_template(human_message_template)
chat_prompt_template = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# Step 3: Read the file content to summarize
file_path = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\2\iac\r\aadb2c_directory.html.markdown'

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    logging.debug("File read successfully.")
except Exception as e:
    logging.error(f"Failed to read file: {e}")
    file_content = "Could not read file content."

# Step 4-5: Format prompt with file content and convert to messages
formatted_prompt = chat_prompt_template.format_prompt(input=f"Please summarize the following content:\n\n{file_content}")
final_prompt_messages = formatted_prompt.to_messages()

# Step 6: Initialize ChatPerplexity and invoke
chat = ChatPerplexity(
    temperature=0,
    pplx_api_key="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",
    model="llama-3.1-sonar-small-128k-online"
)
response = chat.invoke(final_prompt_messages)

# Step 7: Extract and print response
response_content = response.content
print("Summary:", response_content)

# Step 8: Save summary to a file
output_file_path = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\2\iac\iac-agent-1-ouput.txt'

try:
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write("Summary of the document:\n\n")
        output_file.write(response_content)
    logging.debug("Summary saved to output file successfully.")
except Exception as e:
    logging.error(f"Failed to save summary to file: {e}")
