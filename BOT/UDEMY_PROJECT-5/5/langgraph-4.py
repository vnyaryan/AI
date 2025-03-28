import os
import logging
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Step 1: Setup Logging
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\5'
        log_filename = 'langgraph-4.log'

        os.makedirs(log_directory, exist_ok=True)

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='w')
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")

setup_logging()

# Step 2: Define TypedDict Classes
class InputState(TypedDict):
    messages: list[AnyMessage]

# Step 5: Define Node 1 - Collect Input and Print
def node_1(state: InputState) -> InputState:
    """
    Node 1: Collects input from the user, prints it, validates the state, and updates it.
    """
    try:
        logging.debug("Executing node_1 and asking for user input.")
        
        # Prompt the user for input
        user_message = input("You: ")
        
        # Print the user input
        print(f"User Input: {user_message}")
        
        # Append the HumanMessage to the state
        state["messages"].append(HumanMessage(content=user_message))

        logging.debug(f"User message collected and state validated: {state}")
        return state
    except Exception as e:
        logging.error(f"Error in node_1: {e}")
        raise

# Step 5: Define Node 2 - Send Messages to LLM and Retrieve Response
def node_2(state: InputState) -> InputState:
    """
    Node 2: Sends the list of messages to the LLM and retrieves the response.
    """
    try:
        logging.debug(f"Executing node_2 with state: {state}")

        # Initialize the OpenAI model
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

        # Send the messages to the LLM
        llm_response = llm.invoke(state["messages"])

        # Append the AI's response to the messages in the state
        state["messages"].append(AIMessage(content=llm_response.content.strip()))

        logging.debug(f"LLM response received and added to state: {state}")
        return state
    except Exception as e:
        logging.error(f"Error in node_2: {e}")
        raise

# Step 10: Main Execution
if __name__ == "__main__":
    try:
        logging.debug("Starting script execution.")

        # Initialize the InputState
        initial_state = {"messages": []}

        # Execute Node 1
        updated_state = node_1(initial_state)

        # Execute Node 2
        final_state = node_2(updated_state)

        # Print the AI response
        ai_response = final_state["messages"][-1].content
        print(f"AI Response: {ai_response}")

        # Log the final state
        logging.debug(f"Final State: {final_state}")
        print("Script executed successfully.")
    except Exception as e:
        logging.error(f"Error during script execution: {e}")
        raise
