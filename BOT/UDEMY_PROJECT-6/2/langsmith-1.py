import os
import logging
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Literal
import json
import openai
from langsmith import wrappers, traceable


# Logging Setup
def setup_logging():
    try:
        # Define the log directory and filename
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\UDEMY_PROJECT-6\2'
        log_filename = 'langsmith-1.log'

        # Ensure the directory exists
        os.makedirs(log_directory, exist_ok=True)

        # Configure logging
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='w')
        
        # Log the successful setup
        logging.debug("Logging setup completed.")
    except Exception as e:
        # Handle exceptions and print error messages
        print(f"Failed to setup logging: {e}")

# Initialize logging
setup_logging()

# OpenAI API Key
OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# LangSmith API Key
LANGCHAIN_API_KEY = "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"
LANGCHAIN_PROJECT = "pr-vengeful-cattle-49"
LANGCHAIN_TRACING_V2 = "true"





os.environ["OPENAI_API_KEY"] = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "pr-vengeful-cattle-49"



# Set API keys
openai.api_key = OPENAI_API_KEY


# Wrap OpenAI client with LangSmith
client = wrappers.wrap_openai(openai.Client())

@traceable
def pipeline(user_input: str):
    """
    Processes user input using the OpenAI GPT-4 model and logs the response.
    """
    try:
        logging.debug(f"Pipeline called with input: {user_input}")
        result = client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model="gpt-4"
        )
        response = result.choices[0].message.content
        logging.debug(f"Pipeline response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error in pipeline: {e}")
        return "An error occurred while processing your request."

if __name__ == "__main__":
    user_input = input("Enter your message: ").strip()
    if not user_input:
        print("Please enter a valid message.")
    else:
        response = pipeline(user_input)
        print(response)
