import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_models import ChatPerplexity


def setup_logging():
    """Setup logging configuration with a fixed log filename."""
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\1'
        log_filename = 'langchain.log'

        # Ensure the log directory exists
        os.makedirs(log_directory, exist_ok=True)

        # Configure logging
        logging.basicConfig(level=logging.DEBUG,  # Set to DEBUG level
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='a'  # Append mode
        )
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")

# Setup logging
setup_logging()

# Initialize the ChatPerplexity model

chat = ChatPerplexity(
    temperature=0, pplx_api_key="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", model="llama-3.1-sonar-large-128k-online"
)


# Define system and human prompts
system_prompt = "You are devops and terraform expert"
human_prompt = "check URL  and fetch details of Example , Argument Reference , Attributes Reference  this URL - https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/resources/storage_account"

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

# Prepare the messages in the required format for invocation
messages = prompt.format_messages()

# Invoke the chat model using the formatted messages
response = chat.invoke(messages)
print(response.content)