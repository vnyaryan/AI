# Import necessary modules
from dotenv import load_dotenv
from sqlite3 import connect
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import logging

# Load environment variables
load_dotenv()

# Step 1: Setup Logging
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\3'
        log_filename = 'langgraph-5.log'
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=f"{log_directory}/{log_filename}",
            filemode='w',
        )
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")

setup_logging()

# Step 2: Define the State Schema
class State(TypedDict):
    input: str
    user_feedback: str

# Step 3: Define Node Functions
def step_1(state: State) -> None:
    logging.debug(f"---Step 1---, Current State: {state}")
    print("---Step 1---")

def human_feedback(state: State) -> None:
    logging.debug(f"---human_feedback---, Received State: {state}")
    print("---human_feedback---")

def step_3(state: State) -> None:
    logging.debug(f"---Step 3---, Current State: {state}")
    print("---Step 3---")

# Step 4: Build the StateGraph
builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)

# Step 5: Initialize SqliteSaver Checkpointer with Thread Safety
connection = connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(connection)

# Step 6: Compile the Graph with Interruption
graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])

# Step 7: Enhanced Streaming Function
def stream_graph_updates(initial_input: dict, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    logging.debug(f"Initial Input: {initial_input}")
    try:
        for event in graph.stream(initial_input, config, stream_mode="values"):
            if not event:
                logging.debug("Skipping empty event.")
                continue

            logging.debug(f"Event emitted: {event}")
            print(event)
    except Exception as e:
        logging.error(f"An error occurred during streaming: {e}")
        print("An unexpected error occurred during execution.")

# Step 8: Interactive Workflow with User Feedback
if __name__ == "__main__":
    print("Welcome to the Enhanced LangGraph Workflow!")
    thread = {"configurable": {"thread_id": "777"}}

    # Initialize with input state
    initial_input = {"input": "hello world", "user_feedback": ""}
    logging.debug("Starting workflow with initial input.")

    # Stream until interruption
    stream_graph_updates(initial_input, thread["configurable"]["thread_id"])

    # Handle human feedback
    print("The graph is now paused at 'human_feedback'.")
    while True:
        try:
            # Retrieve current state
            state = graph.get_state(thread)
            logging.debug(f"Retrieved StateSnapshot: {state}")
            print("--Current State--")
            print(state)
        except Exception as e:
            logging.error(f"Error retrieving state: {e}")
            print("Failed to retrieve the current state.")
            break

        user_input = input("Tell me how you want to update the state (or type 'quit'): ")
  
