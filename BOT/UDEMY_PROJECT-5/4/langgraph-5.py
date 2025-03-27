import os
import logging
from typing_extensions import TypedDict  # Enables creating dictionary-like data structures with typing.
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver  # For checkpointing in memory.

# Step 1: Setup Logging
def setup_logging():
    """Setup logging configuration with a fixed log filename."""
    try:
        # Define the directory and file for storing logs
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\4'
        log_filename = 'langgraph-5.log'

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

# Define TypedDict Classes
class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    graph_output: str

class OverallState(TypedDict):
    foo: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):
    bar: str

# Define Nodes with Logging
def node_1(state: InputState) -> OverallState:
    try:
        logging.debug(f"Executing node_1 with state: {state}")
        result = {"foo": state["user_input"] + " name"}
        logging.debug(f"node_1 output: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in node_1: {e}")
        raise

def node_2(state: OverallState) -> PrivateState:
    try:
        logging.debug(f"Executing node_2 with state: {state}")
        result = {"bar": state["foo"] + " is"}
        logging.debug(f"node_2 output: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in node_2: {e}")
        raise

def node_3(state: PrivateState) -> OutputState:
    try:
        logging.debug(f"Executing node_3 with state: {state}")
        result = {"graph_output": state["bar"] + " Lance"}
        logging.debug(f"node_3 output: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in node_3: {e}")
        raise

# Build StateGraph
builder = StateGraph(OverallState, input=InputState, output=OutputState)

# Add Nodes
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Add Edges
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

# Setup Checkpointer
checkpointer = MemorySaver()

# Compile Graph with Checkpointer
graph = builder.compile(checkpointer=checkpointer)

# Invoke Graph and Log Results
try:
    logging.debug("Invoking graph with checkpointing...")
    config = {"configurable": {"thread_id": "1"}}  # Configuration for checkpointing
    result = graph.invoke({"user_input": "My"}, config)
    logging.debug(f"Graph output: {result}")
    print(result)  # Output the result for the user

    # Accessing Checkpoint Data
    logging.debug("Accessing checkpoint data...")
    final_state = graph.get_state({"configurable": {"thread_id": "1"}})
    logging.debug(f"Final checkpoint: {final_state}")

    # state_history = graph.get_state_history({"configurable": {"thread_id": "1"}})
    # logging.debug(f"Checkpoint history: {state_history}")

#     checkpoint_history = list(graph.get_state_history({"configurable": {"thread_id": "1"}}))
#     if checkpoint_history:
#         last_checkpoint = checkpoint_history[-1]  # Get the last checkpoint
#         logging.debug(f"Last checkpoint data: {last_checkpoint}")
#     else:
#         logging.debug("No checkpoints available in the state history.")

except Exception as e:
    logging.error(f"Error invoking graph or accessing checkpoints: {e}")

# Retrieve checkpoint history as a generator
checkpoint_history_generator = graph.get_state_history({"configurable": {"thread_id": "1"}})

# Iterate through the generator and log each checkpoint
try:
    for idx, checkpoint in enumerate(checkpoint_history_generator):
        logging.debug(f"Checkpoint {idx + 1}: {checkpoint}")
        print(f"Checkpoint {idx + 1}: {checkpoint}")
except Exception as e:
    logging.error(f"Error while iterating through checkpoint generator: {e}")
