"""
Script to demonstrate the use of LangGraph with a custom reducer.

INPUT:
- The script takes an input dictionary with the following structure:
  {
      "user_input": str  # A user-provided string input
  }

OUTPUT:
- The script outputs a dictionary with the following structure:
  {
      "graph_output": str  # A processed string that combines inputs and outputs of nodes
  }

LOGIC:
1. **Node 1**:
   - Processes the input state and appends " name" to the `user_input` key.
   - Produces an intermediate `OverallState` with fields `foo`, `user_input`, and `graph_output`.

2. **Node 2**:
   - Takes the `OverallState` from Node 1.
   - Generates a new `bar` field in the `PrivateState` by appending " is" to the `foo` value.
   - Applies a custom reducer (`custom_private_state_reducer`) to append the keyword "new" to the `bar` field.

3. **Node 3**:
   - Processes the `PrivateState` from Node 2.
   - Appends " Lance" to the `bar` value to produce the final `graph_output` field in the `OutputState`.

VALIDATION:
- Each node validates its input and output using TypedDict annotations to ensure type correctness.
- Custom validation logic is applied to handle special types like `Annotated`.

CUSTOM REDUCER:
- The custom reducer, `custom_private_state_reducer`, modifies the `bar` field in the `PrivateState` to append the keyword "new".
- This demonstrates how to dynamically alter state values during execution.
-  Reducers in LangGraph are not automatically invoked for sequential updates. They primarily handle state merging during parallel execution or concurrent updates. By explicitly invoking the reducer in node_2, you enforced the desired logic.

USAGE:
- This script demonstrates sequential state updates using LangGraph, showcasing logging, validation, and custom reducer functionality.
"""

import os
import logging
from typing_extensions import TypedDict, Annotated
from typing import get_origin, get_args
from langgraph.graph import StateGraph, START, END

# Step 1: Setup Logging
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\4'
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

# Custom Reducer for PrivateState
def custom_private_state_reducer(existing: dict, new: dict) -> dict:
    """Reducer to append 'new' to the 'bar' key in PrivateState."""
    if "bar" in new:
        new["bar"] = f"{new['bar']} new"
    return new

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
    bar: Annotated[str, custom_private_state_reducer]  # Attach the custom reducer

# Updated Validation Function to Handle Annotated Types
def validate_typed_dict(data: dict, expected_type: TypedDict):
    for key, value_type in expected_type.__annotations__.items():
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in data: {data}")
        
        # Handle Annotated types by extracting the base type
        base_type = get_origin(value_type) or value_type
        if base_type is Annotated:
            base_type = get_args(value_type)[0]

        if not isinstance(data[key], base_type):
            raise TypeError(f"Key '{key}' has incorrect type: expected {value_type}, got {type(data[key])}")
    logging.info(f"Validation successful for data: {data}")

# Specific Validation Functions
def validate_input_state(data: dict):
    validate_typed_dict(data, InputState)

def validate_overall_state(data: dict):
    validate_typed_dict(data, OverallState)

def validate_private_state(data: dict):
    validate_typed_dict(data, PrivateState)

def validate_output_state(data: dict):
    validate_typed_dict(data, OutputState)

# Define Nodes with Logging and Validation
def node_1(state: InputState) -> OverallState:
    try:
        logging.debug(f"Executing node_1 with state: {state}")
        validate_input_state(state)  # Validate input
        result = {"foo": state["user_input"] + " name", "user_input": state["user_input"], "graph_output": ""}
        validate_overall_state(result)  # Validate output
        logging.debug(f"node_1 output: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in node_1: {e}")
        raise

def node_2(state: OverallState) -> PrivateState:
    try:
        logging.debug(f"Executing node_2 with state: {state}")
        validate_overall_state(state)  # Validate input
        
        # Generate initial result
        result = {"bar": state["foo"] + " is"}
        
        # Explicitly call the reducer to modify the result
        result = custom_private_state_reducer({}, result)
        
        validate_private_state(result)  # Validate output
        logging.debug(f"node_2 output: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in node_2: {e}")
        raise


def node_3(state: PrivateState) -> OutputState:
    try:
        logging.debug(f"Executing node_3 with state: {state}")
        validate_private_state(state)  # Validate input
        result = {"graph_output": state["bar"] + " Lance"}
        validate_output_state(result)  # Validate output
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
