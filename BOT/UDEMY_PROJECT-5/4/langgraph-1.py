"""
This script demonstrates the use of the `langgraph` library to create and execute a state graph workflow. 
The script performs the following tasks:
1. Sets up logging to capture detailed runtime information in a specified log file.
2. Defines structured state management using `TypedDict` for type-safe input, intermediate, and output states.
3. Implements three graph nodes (`node_1`, `node_2`, and `node_3`) to sequentially process input data:
   - `node_1`: Appends " name" to the user input.
   - `node_2`: Appends " is" to the result of `node_1`.
   - `node_3`: Appends " Lance" to the result of `node_2`.
4. Constructs a state graph with defined edges to establish the workflow from START to END.
5. Compiles and invokes the graph with an initial input (`user_input`) and outputs the final result.
6. Includes comprehensive error handling and logging for all stages of the graph execution.

Final Output Example:
For input `{"user_input": "My"}`, the graph produces `{"graph_output": "My name is Lance"}`.
"""


import os
import logging
from typing_extensions import TypedDict  # Enables creating dictionary-like data structures with typing.

from langgraph.graph.message import add_messages  # Helper to manage message structures in the state graph.
from langgraph.graph import StateGraph, START, END

# Step 1: Setup Logging
def setup_logging():
    """Setup logging configuration with a fixed log filename."""
    try:
        # Define the directory and file for storing logs
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\4'
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

# Compile Graph
graph = builder.compile()

# Invoke Graph and Log Results
try:
    logging.debug("Invoking graph...")
    result = graph.invoke({"user_input": "My"})
    logging.debug(f"Graph output: {result}")
    print(result)  # Output the result for the user
except Exception as e:
    logging.error(f"Error invoking graph: {e}")
