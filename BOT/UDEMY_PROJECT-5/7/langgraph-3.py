import os
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# Setup the OpenAI API Key
OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Logging setup function
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\7'
        log_filename = 'langgraph-3.log'

        os.makedirs(log_directory, exist_ok=True)

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='w')
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        raise

setup_logging()

# Define the tools for the agent to use
@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

tools = [search]

tool_node = ToolNode(tools)

# Initialize the ChatOpenAI model
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# Define the function that determines whether to continue or go to node3
def should_continue(state: MessagesState) -> Literal["tools", "node3"]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we route to "node3"
    return "node3"

# Define the function that calls the model (used in node3)
def node3(state: MessagesState) -> MessagesState:
    """
    Node 3: Sends the conversation history to the LLM for a final response.
    """
    try:
        messages = state['messages']
        logging.debug("Processing node3...")
        response = model.invoke(messages)
        logging.debug(f"LLM Final Response: {response.content}")
        print(f"LLM Final Response: {response.content}")  # Print the response to the user
        # Add the response to the state
        state["messages"].append(response)
        return state
    except Exception as e:
        logging.error(f"Error in node3: {e}")
        raise

# Define the function that calls the model (used in the agent node)
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define a new graph
workflow = StateGraph(MessagesState)

# Define the nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("node3", node3)  # Add node3

# Set the entrypoint as `agent`
workflow.add_edge(START, "agent")

# Add a conditional edge from "agent" to either "tools" or "node3"
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

# Add an edge from "tools" to "agent"
workflow.add_edge("tools", "agent")

# Compile the graph with a memory saver
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="temperature in San Francisco")]},
    config={"configurable": {"thread_id": 42}}
)
