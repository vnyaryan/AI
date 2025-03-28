"""
Script Logic:
1. **Setup Logging**:
   - Creates a log directory if it doesn't exist.
   - Configures logging to write logs into a file (`langgraph-2.log`) for debugging and monitoring.

2. **Define TypedDict Classes**:
   - `InputState`: Represents the structure of the input data to the graph.
   - `OutputState`: Represents the structure of the final output data from the graph.
   - `OverallState`: Used to define the intermediate state between nodes in the graph.
   - `PrivateState`: Used for intermediate processing in specific nodes.

3. **Validation Functions**:
   - Generic `validate_typed_dict`: Validates that a dictionary adheres to a specific TypedDict schema.
   - Specific functions (`validate_input_state`, `validate_overall_state`, etc.) wrap the generic validator for each TypedDict.

4. **Define Graph Nodes**:
   - `node_1`: Takes `InputState`, processes it, and produces `OverallState`. Appends " name" to the `user_input`.
   - `node_2`: Takes `OverallState`, processes it, and produces `PrivateState`. Appends " is" to the `foo` value.
   - `node_3`: Takes `PrivateState`, processes it, and produces `OutputState`. Appends " Lance" to the `bar` value.

5. **Build StateGraph**:
   - Creates a `StateGraph` that uses the defined TypedDict schemas for input, intermediate, and output states.
   - Nodes are added sequentially to the graph (`node_1`, `node_2`, `node_3`).
   - Directed edges define the flow of data between nodes.

6. **Compile and Invoke Graph**:
   - Compiles the graph using `builder.compile()`.
   - Invokes the graph with the initial input (`{"user_input": "My"}`).
   - Logs the results and outputs the final result.

7. **Error Handling**:
   - Includes try-except blocks in all nodes and the graph invocation process to catch and log errors, ensuring robust debugging.

Key Features:
- Logging ensures traceability of the execution process.
- Validation enforces strict adherence to the expected data structures.
- Modular nodes enable easy scalability and maintenance of the graph.
"""



import os
import logging
from typing_extensions import TypedDict 
from langgraph.graph import StateGraph, START, END

# Step 1: Setup Logging
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\4'
        log_filename = 'langgraph-2.log'

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

# General Validation Function
def validate_typed_dict(data: dict, expected_type: TypedDict):
    for key, value_type in expected_type.__annotations__.items():
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in data: {data}")
        if not isinstance(data[key], value_type):
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
        result = {"bar": state["foo"] + " is"}
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
