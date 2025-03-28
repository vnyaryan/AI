# This script implements a LangChain-based chatbot using LangGraph, a graph-based state management tool.
# The script is divided into the following steps:
# 1. Setup Logging: Configures logging to record events and errors into a specified log file.
# 2. OpenAI API Setup: Initializes the ChatOpenAI model using the provided API key for chatbot interactions.
# 3. Define State and Build StateGraph: Constructs the graph of states for managing chatbot responses.
# 4. Define Streaming Function: Implements a function to stream updates from the chatbot graph.
# 5. Interactive Conversation Loop: Provides an interactive interface for the user to chat with the chatbot.

import os
import logging
from langchain_openai import ChatOpenAI  # LangChain's ChatOpenAI module for AI interactions.
from typing import Annotated  # Used for type annotations with additional metadata.
from typing_extensions import TypedDict  # Enables creating dictionary-like data structures with typing.
from langgraph.graph import StateGraph  # Manages state transitions in the chatbot.
from langgraph.graph.message import add_messages  # Helper to manage message structures in the state graph.

# Step 1: Setup Logging
def setup_logging():
    """Setup logging configuration with a fixed log filename."""
    try:
        # Define the directory and file for storing logs
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\3'
        log_filename = 'langgraph-1.log'

        # Ensure the log directory exists
        os.makedirs(log_directory, exist_ok=True)

        # Configure logging with DEBUG level to capture detailed information
        logging.basicConfig(level=logging.DEBUG,  
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='w')  # Overwrite the log file for each run.
        logging.debug("Logging setup completed.")  # Log a success message.
    except Exception as e:
        print(f"Failed to setup logging: {e}")  # Print an error message if logging setup fails.

setup_logging()

# Step 2: OpenAI API Setup
# Define the OpenAI API key and initialize the ChatOpenAI model for natural language interactions.
OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# Step 3: Define State and Build StateGraph
# A TypedDict representing the state structure for the graph.
# A schema defined using TypedDict specifies a dictionary-like structure where each key is associated with a specific value type
#  State is name of class  
# typedict  is used to create dictionary-like structure of schema  which contain  key value pair and value represent  type of key
# messages  represent key 
#  list represent type of key 
# add_message is reducer function 
# Annotated  is used to combine   list with reducer function 
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
#  SYNTAX 
#  class <name of class>(TypedDict):
#        <name of key>: Annotated[value of key representing type ,  reducer function]        
class State(TypedDict):
    messages: Annotated[list, add_messages]  # A list of messages annotated with helper metadata.

# Create a StateGraph object to manage transitions and logic in the chatbot's workflow.
# StateGraph is a name of class  and it will accept 2 parameter state_schema , config_schema (OPTIONAL)
# https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph
# SYNTAX
# StateGraph(NAME OF state_schema)
graph_builder = StateGraph(State)

# Define the chatbot function that processes the state and generates a response.
# CREATE NODE  -  NODE IS JUST A PYTHON FUNCTION
# NAME OF FUNCTION IS NAME OF  NODE
#  chatbot is defined to accept a single parameter, state, which is an instance of the State class.
#  llm.invoke(state["messages"]) calls the invoke method of the llm
#  By passing the conversation history (state["messages"]) to this method, the language model generates a context-aware response.
#  The function returns a dictionary with a single key, "messages", whose value is a list containing the response from the language model.
#  this function rturn a dictionary with a single key, "messages" -  and this be passed to State  which  has reducer function add_message and this 
#  add_message is used to  list containing the response from the language model  ( return by llm)  to the existing message  

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}  



# Add the chatbot function as a node in the graph.
# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

# Set the starting and ending points of the graph to the chatbot node.
# add an entry point. This tells our graph where to start its work each time we run it.
graph_builder.set_entry_point("chatbot")

# set a finish point. This instructs the graph "any time this node is run, you can exit."
graph_builder.set_finish_point("chatbot")

# Compile the graph for execution.
graph = graph_builder.compile()

# Step 4: Define Streaming Function
# This function streams updates from the chatbot graph and outputs responses to the user.
def stream_graph_updates(user_input: str):
    # Stream events from the graph based on user input.
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():  # Extract response values from the streamed event.
            print("Assistant:", value["messages"][-1].content)  # Print the latest message content.

# Step 5: Interactive Conversation Loop
# Run the chatbot in an interactive loop where the user can provide inputs.
if __name__ == "__main__":
    print("Welcome to the LangGraph Chatbot! Type 'quit' to exit the conversation.")  # Welcome message.
    while True:
        try:
            # Prompt the user for input
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:  # Exit conditions.
                print("Goodbye!")
                break

            # Pass the user input to the streaming function to process and respond.
            stream_graph_updates(user_input)
        except Exception as e:
            logging.error(f"An error occurred: {e}")  # Log any errors that occur.
            print("An unexpected error occurred. Ending conversation.")  # Inform the user about the error.
            break
