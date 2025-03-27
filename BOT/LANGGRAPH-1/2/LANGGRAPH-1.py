import os
import logging
from langgraph.graph.message import add_messages
from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from typing_extensions import TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Constants
PINECONE_API_KEY = "9fe3c832-5827-45f6-b066-e740b9b13e33"
PINECONE_ENVIRONMENT = "us-east-1"
TEST_INDEX1 = "testindex1"
API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# System Prompt
system_prompt = """
You are an intelligent assistant.

you will check each user input and respond accordingly and in case more informatoion required then you will ask for more  information.
"""

def setup_logging():
    """
    Sets up logging for the application.

    Creates the log directory if it does not exist and configures the logging settings 
    to save logs in the specified file with detailed debug information.

    Raises:
        Exception: If there's an error while setting up logging.
    """
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\LANGGRAPH-1\2'
        log_filename = 'langgraph-1.log'

        os.makedirs(log_directory, exist_ok=True)

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='w')
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        raise

def write_all_messages_to_file(state_values, output_file="all_msg.txt"):
    """
    Writes MessagesState details to a text file for auditing and debugging.

    Args:
        state_values (dict): The current state of the workflow.
        output_file (str): Path to the file where details will be logged.
    """
    try:
        with open(output_file, "a") as file:
            file.write("======== START OF ITERATION ========\n")
            file.write(json.dumps(
                {"messages": [message.__dict__ for message in state_values["messages"]]},
                indent=4
            ))
            file.write("\n======== END OF ITERATION ========\n\n")
        logging.debug(f"Messages state successfully logged to {output_file}.")
    except Exception as e:
        logging.error(f"Error writing messages to file: {e}")
        raise

# Function to write MessagesState details to a text file
def write_all_message_types_to_txt(state_values, output_file="msg_details.txt"):
    """
    Processes and writes the content of all message types (HumanMessage, AIMessage, ToolMessage, etc.)
    to a text file with separators
    """
    try:
        with open(output_file, "w") as file:
            file.write("======== START OF MESSAGES ========\n\n")
            for message_group in state_values:
                for message in message_group:
                    if isinstance(message, HumanMessage):
                        file.write(f"HumanMessage Content: {message.content}\n")
                        file.write("-------------------------------------------------------------------\n")
                    elif isinstance(message, AIMessage):
                        file.write(f"AIMessage Content: {message.content}\n")
                        file.write("-------------------------------------------------------------------\n")
                    elif isinstance(message, SystemMessage):
                        file.write(f"SystemMessage Content: {message.content}\n")
                        file.write("-------------------------------------------------------------------\n") 
                    elif isinstance(message, ToolMessage):
                        file.write(f"ToolMessage Content: {message.content}\n")
                        file.write(f"  Tool Name: {message.name}\n")
                        file.write("-------------------------------------------------------------------\n")
            file.write("\n======== END OF MESSAGES ========\n")
        logging.debug(f"All message types successfully written to {output_file}")
    except Exception as e:
        logging.error(f"Error writing message types to file: {e}")



def initialize_pinecone():
    """
    Initializes Pinecone with the provided API key and environment and connects to the specified index.

    Returns:
        pinecone.Index: Initialized Pinecone index.

    Raises:
        Exception: If there's an error during Pinecone initialization.
    """
    try:
        logging.debug("Initializing Pinecone.")

        # Create Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Check if the index exists, create if it doesn't
        if TEST_INDEX1 not in pc.list_indexes().names():
            logging.debug(f"Creating Pinecone index '{TEST_INDEX1}'.")
            pc.create_index(
                name=TEST_INDEX1,
                dimension=1536,  # Adjust dimension to match your embeddings
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",  # Adjust cloud provider if necessary
                    region=PINECONE_ENVIRONMENT
                )
            )

        # Connect to the index
        index = pc.Index(TEST_INDEX1)
        logging.debug(f"Pinecone index '{TEST_INDEX1}' initialized successfully.")
        return index
    except Exception as e:
        logging.error(f"Error initializing Pinecone: {e}")
        raise


def node1(state: MessagesState):
    """
    Node 1: Collects a single user input and adds it to the state.
    """
    try:
        logging.debug("NODE1 EXECUTION STARTED.")
        
        # Prompt the user for input
        user_message = input("You: ")

        # Check for termination condition
        if user_message.lower() == "exit":
            logging.debug("Termination condition met in Node 1. Exiting conversation.")
            print("Conversation ended successfully.")
            logging.debug("NODE1 EXECUTION COMPLETED.")
            exit(0)  

        # Add the user message to the state using add_messages
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_message)]) 

        # Log details
        last_message = state["messages"][-1]
        logging.debug(f"User message added to state: {last_message.content}")
        logging.debug("NODE1 EXECUTION COMPLETED.")
        return state
    except Exception as e:
        logging.error(f"Error in node_1: {e}")
        raise

def define_graph():
    """
    Defines the workflow graph with nodes and edges.

    Returns:
        StateGraph: A compiled graph representing the workflow.
    """
    logging.debug("Defining the workflow graph.")

    # Create the graph builder
    builder = StateGraph(MessagesState)

    # Add nodes
    builder.add_node("user_query", node1)
    # builder.add_node("query_embedding", query_embedding_node)
    # builder.add_node("similarity_search", similarity_search_node)
    # builder.add_node("reflection", reflection_node)
    # builder.add_node("query_transformation", query_transformation_node)
    # builder.add_node("context_refinement", context_refinement_node)
    # builder.add_node("generation", generation_node)
    # builder.add_node("self_reflection", self_reflection_node)
    # builder.add_node("end", end_node)

    # Add edges to define the workflow
    builder.add_edge(START, "user_query")
    # builder.add_edge("user_query", "query_embedding")
    # builder.add_edge("query_embedding", "similarity_search")
    # builder.add_edge("similarity_search", "reflection")
    # builder.add_edge("reflection", "query_transformation")
    # builder.add_edge("query_transformation", "context_refinement")
    # builder.add_edge("context_refinement", "generation")
    # builder.add_edge("generation", "self_reflection")
    # builder.add_edge("self_reflection", "end")




    # Initialize Memory Saver
    memory_saver = MemorySaver()
    logging.debug("Memory saver initialized.")

    # Compile graph with interrupt handling
    graph = builder.compile(checkpointer=memory_saver)
    logging.debug("Workflow graph compiled successfully.")
    return graph




def main():
    """
    Main function to execute the workflow graph.
    """
    try:
        # Setup logging
        setup_logging()
                
        logging.debug("Starting multi-turn conversation.")
        initial_state = {"messages": [SystemMessage(content=system_prompt)]}
        iteration = 0
        


        # Define and compile the graph
        graph = define_graph()
        
        # Initialize LLM and Pinecone and embeddings model

        pinecone_index = initialize_pinecone()
        embeddings_model = OpenAIEmbeddings(api_key=API_KEY)


        while True:
            iteration += 1
            logging.debug(f"START ITERATION - {iteration}")

            # Invoke the graph
            updated_state = graph.invoke(initial_state, config={"configurable": {"thread_id": 1}})
            ai_response = updated_state["messages"][-1].content
            print(f"AI: {ai_response}")

            # Log messages state to file
            write_all_messages_to_file(updated_state)

            initial_state = updated_state
            
            # Write the messages to the specified file
            write_all_message_types_to_txt(updated_state.values())
            logging.debug(f"ENDING ITERATION - {iteration}")
            
    except Exception as e:
        logging.error(f"Error during multi-turn conversation: {e}")
        raise

if __name__ == "__main__":
    main()
