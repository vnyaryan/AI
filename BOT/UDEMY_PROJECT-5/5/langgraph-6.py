import os
import logging
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver for memory handling

OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Step 1: Setup Logging
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\5'
        log_filename = 'langgraph-6.log'

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

class FinalState(TypedDict):
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
def node_2(state: InputState) -> FinalState:
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

# Step 6: Build StateGraph
builder = StateGraph(
    FinalState,
    input=InputState,
    output=FinalState
)

# Add Nodes
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)

# Add Edges
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

# Initialize Memory Saver
prompt_memory_saver = MemorySaver()

# Compile Graph with MemorySaver
graph = builder.compile(checkpointer=prompt_memory_saver)

# Step 7: Invoke Graph and Log Results
if __name__ == "__main__":
    try:
        logging.debug("Starting graph invocation.")
        
        # Initialize input state
        initial_state = {"messages": []}
        
        # Configuration for memory checkpointer
        config = {"configurable": {"thread_id": "1"}}
        
        # Invoke the graph
        result = graph.invoke(initial_state, config)
        
        # Print the final AI response
        ai_response = result["messages"][-1].content
        print(f"AI Response: {ai_response}")
        
        # Log the final state
        logging.debug(f"Final State: {result}")
        print("Script executed successfully.")
        
        # Access and log checkpoint history
        logging.debug("Accessing checkpoint history...")
        checkpoint_history_generator = graph.get_state_history(config)
        for idx, checkpoint in enumerate(checkpoint_history_generator):
            logging.debug(f"Checkpoint {idx + 1}: {checkpoint}")
        
        # Access the final checkpoint state
        final_checkpoint_state = graph.get_state(config)
        logging.debug(f"Final checkpoint state: {final_checkpoint_state}")

    except Exception as e:
        logging.error(f"Error during graph invocation: {e}")
        raise
