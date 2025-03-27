import os
import logging
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
import random



# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Define node functions
def node_a(state: dict) -> Command:
    """
    Node A: Updates the state directly and routes to node_b or node_c.
    """
    logging.debug("Executing Node A")
    value = random.choice(["a", "b"])
    logging.debug(f"Value chosen in Node A: {value}")

    # Update state directly
    state["key_from_a"] = value

    # Route to node_b if value is "a", else to node_c
    goto = "node_b" if value == "a" else "node_c"
    return Command(goto=goto)


def node_b(state: dict) -> Command:
    """
    Node B: Updates the state directly and routes to node_c.
    """
    logging.debug("Executing Node B")

    # Update state directly
    state["key_from_b"] = "Value from B"
    return Command(goto="node_c")


def node_c(state: dict) -> dict:
    """
    Node C: Final node, updates the state directly and ends.
    """
    logging.debug("Executing Node C")

    # Update state directly
    state["key_from_c"] = "Value from C"
    logging.debug(f"Final State: {state}")
    return state


# Define the graph
def define_graph():
    """
    Define a simple graph with three nodes.
    """
    builder = StateGraph(MessagesState)

    # Add nodes
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_node("node_c", node_c)

    # Add edges
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_a", "node_c")  # Add both edges for conditional routing
    builder.add_edge("node_b", "node_c")

    # Compile graph
    return builder.compile()


# Main script
if __name__ == "__main__":
    logging.debug("Defining the test graph...")
    graph = define_graph()

    # Initial state
    initial_state = {"messages": []}

    # Invoke the graph
    logging.debug("Starting the graph execution...")
    updated_state = graph.invoke(initial_state)
    logging.debug(f"Updated State after execution: {updated_state}")
