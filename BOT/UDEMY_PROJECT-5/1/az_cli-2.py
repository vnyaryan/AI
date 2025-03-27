
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


setup_logging()


chat = ChatPerplexity(
    temperature=0, pplx_api_key="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", model="llama-3.1-sonar-small-128k-online"
)



system = "You are a helpful assistant."
human = "{input}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
response = chain.invoke({"input": "Why is the Higgs Boson important?"})
response.content