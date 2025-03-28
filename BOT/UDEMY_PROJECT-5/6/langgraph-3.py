import os
import logging
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages  # Import add_messages


OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Step 1: Setup Logging
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\6'
        log_filename = 'langgraph-3.log'

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
class OverallState(TypedDict):
    messages: list[AnyMessage]

# Step 3: Define Node 1 - Collect One Input
def node_1(state: OverallState) -> OverallState:
    """
    Node 1: Collects a single user input and adds it to the state.
    """
    try:
        logging.debug("Executing node_1 and collecting user input.")
        
        # Prompt the user for input
        user_message = input("You: ")
        
        # Check for termination condition
        if user_message.lower() == "exit":
            print("Ending conversation.")
            return None
        
        # Add the user message to the state using add_messages
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_message)])  # Assign the result
        logging.debug(f"User message added to state via add_messages: {state['messages']}")
        
        return state
    except Exception as e:
        logging.error(f"Error in node_1: {e}")
        raise

# Step 4: Define Node 2 - Send Messages to LLM and Retrieve Response
def node_2(state: OverallState) -> OverallState:
    """
    Node 2: Sends the conversation history to the LLM and appends the AI's response.
    """
    try:
        logging.debug(f"Executing node_2 with state: {state}")

        # Initialize the OpenAI model
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

        # Ensure messages array is not empty
        if not state["messages"]:
            raise ValueError("State messages are empty. Cannot send an empty conversation to the LLM.")

        # Send the messages to the LLM
        llm_response = llm.invoke(state["messages"])

        # Add the AI's response to the state using add_messages
        state["messages"] = add_messages(state["messages"], [AIMessage(content=llm_response.content.strip())])  # Assign the result
        logging.debug(f"LLM response added to state via add_messages: {state['messages']}")
        
        return state
    except Exception as e:
        logging.error(f"Error in node_2: {e}")
        raise

# Step 5: Define Node 3 - Use add_two_numbers Tool

def node_3(state: OverallState) -> OverallState:
    """
    Node 3: Collects two numbers from the user, uses the add_two_numbers tool,
    and adds the result to the conversation.
    """
    try:
        logging.debug("Executing node_3 to add two numbers.")
        
        # Collect numbers from the user
        num1 = float(input("Enter the first number: "))
        num2 = float(input("Enter the second number: "))
        
        # Use the tool
        result = add_two_numbers.invoke({"num1": num1, "num2": num2})  # Use invoke instead of calling directly
        
        # Add the result to the state as an AIMessage
        state["messages"] = add_messages(state["messages"], [AIMessage(content=f"The sum is: {result}")])
        logging.debug(f"Result of add_two_numbers added to state: {state['messages']}")
        
        return state
    except Exception as e:
        logging.error(f"Error in node_3: {e}")
        raise


# Step 6: Build StateGraph
builder = StateGraph(
    OverallState,
    input=OverallState,
    output=OverallState
)

# Add Nodes
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Add Edges
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

# Initialize Memory Saver
prompt_memory_saver = MemorySaver()

# Compile Graph with MemorySaver
graph = builder.compile(checkpointer=prompt_memory_saver)

# Step 7: Invoke Graph and Log Results
if __name__ == "__main__":
    try:
        logging.debug("Starting multi-turn conversation.")
        
        # Initialize input state
        initial_state = {"messages": []}
        config = {"configurable": {"thread_id": "1"}}

        while True:
            # Collect user input
            updated_state = node_1(initial_state)

            # Check if conversation ended
            if updated_state is None:
                break

            # Send messages to LLM and get a response
            final_state = node_2(updated_state)

            # Use the add_two_numbers tool
            final_state = node_3(final_state)

            # Print the AI response
            ai_response = final_state["messages"][-1].content
            print(f"AI: {ai_response}")
            
            # Update the state for the next turn
            initial_state = final_state

        logging.debug(f"Final State: {initial_state}")
        print("Conversation ended successfully.")
    except Exception as e:
        logging.error(f"Error during multi-turn conversation: {e}")
        raise
