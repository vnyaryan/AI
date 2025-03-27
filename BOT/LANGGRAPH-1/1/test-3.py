import logging
from typing import List, Dict
from langchain_openai.embeddings import OpenAIEmbeddings
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
from typing_extensions import TypedDict, Literal

# Mock API Key for testing
API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Define the EmbeddingState class
class EmbeddingState:
    """
    A custom class to manage embedding-related state with strict key-value types.
    """

    def __init__(self):
        """
        Initialize the embedding state as an empty dictionary.
        """
        self.data: Dict[str, Dict] = {}

    def add_embedding(self, key: str, query: str, embedding: List[float]):
        """
        Add embedding details to the state.

        Args:
            key (str): The key (e.g., user query or identifier).
            query (str): The original user query.
            embedding (List[float]): The embedding vector.
        """
        self.data[key] = {
            "embedding": embedding,
            "dimensions": len(embedding),
            "query": query,
        }

    def get_embedding(self, key: str) -> Dict:
        """
        Retrieve embedding details by key.

        Args:
            key (str): The key to look up.

        Returns:
            Dict: The embedding details if found, otherwise raises KeyError.
        """
        return self.data[key]

    def to_dict(self) -> Dict[str, Dict]:
        """
        Convert the state to a dictionary.

        Returns:
            Dict[str, Dict]: The state as a dictionary.
        """
        return self.data


# Define the process_query_embedding function
def process_query_embedding(state, user_query, api_key):
    """
    Generate and store embeddings for a user query.

    Args:
        state (dict): The current workflow state.
        user_query (str): The user's query.
        api_key (str): The OpenAI API key.

    Returns:
        None: Updates the `embedding_state` within the workflow state.
    """
    try:
        logging.debug("PROCESS QUERY EMBEDDING EXECUTION STARTED.")

        # Initialize the embeddings model
        embeddings_model = OpenAIEmbeddings(api_key=api_key)

        # Generate the embedding
        embedding = embeddings_model.embed_documents([user_query])[0]
        logging.debug(f"Generated embedding for query: '{user_query}' with dimensions {len(embedding)}")

        # Initialize the embedding state if not present
        if "embedding_state" not in state:
            state["embedding_state"] = EmbeddingState()
            logging.debug("Initialized embedding state.")

        # Add the embedding to the embedding state
        state["embedding_state"].add_embedding(key="query_1", query=user_query, embedding=embedding)
        logging.debug("Updated embedding state with new embedding.")

        logging.debug("PROCESS QUERY EMBEDDING EXECUTION COMPLETED.")
    except Exception as e:
        logging.error(f"Error in PROCESS QUERY EMBEDDING: {e}")
        raise


# Test the EmbeddingState class and process_query_embedding function
def test_embedding_state():
    """
    Test the EmbeddingState class and process_query_embedding function.
    """
    try:
        # Setup logging
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

        # Initialize a mock state
        state = {
            "messages": [{"content": "What is the weather today?"}]
        }

        # Retrieve the user query
        user_query = state["messages"][-1]["content"]

        # Call the process_query_embedding function
        process_query_embedding(state, user_query, API_KEY)

        # Validate the embedding state
        embedding_state = state["embedding_state"]
        assert embedding_state is not None, "Embedding state was not initialized."
        assert embedding_state.get_embedding("query_1") is not None, "Embedding was not added to the state."

        # Print results
        print("Test Passed!")
        print("Embedding State:")
        for key, value in embedding_state.to_dict().items():
            print(f"Key: {key}, Value: {value}")
    except Exception as e:
        print(f"Test Failed: {e}")


# Run the test
if __name__ == "__main__":
    test_embedding_state()
