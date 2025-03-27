import os
import logging
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver for checkpointing.

# Step 1: Setup Logging
def setup_logging():
    """Setup logging configuration with a fixed log filename."""
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\4'
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

# Enable Checkpointer and Breakpoint
memory = MemorySaver()  # Initialize the MemorySaver checkpointer.
graph = builder.compile(checkpointer=memory, interrupt_before=["node_3"])

# Invoke Graph with Breakpoint Logic
try:
    logging.debug("Invoking graph...")
    for event in graph.stream({"user_input": "My"}, {"configurable": {"thread_id": "1"}}, stream_mode="values"):
        print(event)  # Print events up to the breakpoint.

    # Custom Logic to Handle Breakpoint
    user_input = input("Do you want to proceed to node_3? (yes/no): ").strip().lower()
    if user_input == "yes":
        logging.debug("User approved. Resuming graph execution...")
        for event in graph.stream(None, {"configurable": {"thread_id": "1"}}, stream_mode="values"):
            print(event)
    else:
        logging.debug("User declined. Halting execution.")
        print("Graph execution stopped.")
except Exception as e:
    logging.error(f"Error invoking graph: {e}")
