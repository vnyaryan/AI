import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationSummaryMemory

def setup_logging():
    """Setup logging configuration with a fixed log filename."""
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\13'
        log_filename = 'langchain.log'

        # Ensure the log directory exists
        os.makedirs(log_directory, exist_ok=True)

        # Configure logging
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='a'  # Append mode
        )
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")

# Step 1: Setup logging
setup_logging()

# Step 2: Securely manage the API key and initialize the ChatOpenAI class with it.
api_key = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"  # Replace with your actual API key.

# Step 3: Initialize conversation summary memory
memory = ConversationSummaryMemory(
    memory_key="messages",   
    return_messages=True,    
    llm=ChatOpenAI(openai_api_key=api_key)  
)

llm = ChatOpenAI(
    openai_api_key=api_key  
)
logging.debug("ChatOpenAI initialized with provided API key.")

# Step 4: Define system and human message templates for the first prompt (Azure CLI command)

system_message_template = """
You are an expert in Azure CLI. Based on the user's query, identify the Azure CLI command and list all required 
and optional parameters.
"""

human_message_template = "{task}"

system_message_prompt_template = SystemMessagePromptTemplate.from_template(system_message_template)
human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_message_template)

chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt_template, human_message_prompt_template]
)

try:
    # Step 5: Generate the Azure CLI command and parameters using the first chain
    task_query = "how to creeate  azure kubernetes"
    final_prompt = chat_prompt_template.format_prompt(task=task_query).to_messages()
    logging.debug("Final prompt created for first chain: %s", final_prompt)

    llm_response = llm.invoke(final_prompt)
    response_content = llm_response.content
    logging.debug("LLM response received for first chain: %s", response_content)

    # Save the response to memory
    memory.save_context({'input': final_prompt}, {'output': response_content})

    # Step 6: Define second chain for formatting the output
    system_message_template_format = "Act as expert of summarize all details ,You have to summarize all details: {cli_output}"

    system_message_prompt_template_format = SystemMessagePromptTemplate.from_template(system_message_template_format)
    human_message_prompt_template_format = HumanMessagePromptTemplate.from_template("{cli_output}")

    chat_prompt_template_format = ChatPromptTemplate.from_messages(
        [system_message_prompt_template_format, human_message_prompt_template_format]
    )

    # Step 7: Format the output generated from the first chain using the second chain
    formatted_prompt = chat_prompt_template_format.format_prompt(cli_output=response_content).to_messages()
    logging.debug("Formatted prompt created for second chain: %s", formatted_prompt)

    formatted_response = llm.invoke(formatted_prompt)
    logging.debug("LLM formatted response received for second chain: %s", formatted_response.content)

    # Save the formatted response to memory
    memory.save_context({'input': formatted_prompt}, {'output': formatted_response.content})

    # Step 8: Output the results
    print("Generated CLI Command and Parameters:")
    print(response_content)
    print("\nFormatted Output:")
    print(formatted_response.content.strip())

except Exception as e:
    logging.error("An error occurred: %s", e)
    print(f"An error occurred: {e}")
