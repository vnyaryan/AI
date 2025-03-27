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
from typing_extensions import TypedDict, Literal

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Define graph state using TypedDict
class WorkflowState(TypedDict):
    key_from_a: str
    key_from_b: str
    key_from_c: str


# Define node functions
def node_a(state: WorkflowState) -> Command[Literal["node_b", "node_c"]]:
    """
    Node A: Updates the state and routes to node_b or node_c.
    """
    logging.debug("Executing Node A")
    value = random.choice(["a", "b"])
    logging.debug(f"Value chosen in Node A: {value}")

    # Update the state and route to the next node
    return Command(
        update={"key_from_a": value},
        goto="node_b" if value == "a" else "node_c"
    )


def node_b(state: WorkflowState) -> Command[Literal["node_c"]]:
    """
    Node B: Updates the state and routes to node_c.
    """
    logging.debug("Executing Node B")

    # Update the state and route to the next node
    return Command(
        update={"key_from_b": "Value from B"},
        goto="node_c"
    )


def node_c(state: WorkflowState) -> Command[Literal[END]]:
    """
    Node C: Final node, updates the state and ends.
    """
    logging.debug("Executing Node C")

    # Update the state and end the workflow
    return Command(
        update={"key_from_c": "Value from C"},
        goto=END
    )


# Define the graph
def define_graph():
    """
    Define a simple graph with three nodes and no explicit edges between nodes.
    """
    builder = StateGraph(WorkflowState)

    # Add nodes
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_node("node_c", node_c)

    # Add entry point
    builder.add_edge(START, "node_a")

    # Compile graph
    return builder.compile()


# Main script
if __name__ == "__main__":
    logging.debug("Defining the test graph...")
    graph = define_graph()

    # Initial state
    initial_state: WorkflowState = {"key_from_a": "", "key_from_b": "", "key_from_c": ""}

    # Invoke the graph
    logging.debug("Starting the graph execution...")
    updated_state = graph.invoke(initial_state)
    logging.debug(f"Updated State after execution: {updated_state}")
