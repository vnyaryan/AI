"""
Script Overview:
----------------
This script implements a modular workflow for processing user queries, generating embeddings,
performing similarity searches, refining queries, and generating responses using Pinecone 
and OpenAI's LLM. The workflow is built as a graph, where each node represents a specific 
processing step, and the state transitions between nodes are managed seamlessly.

Workflow Logic:
---------------
1. **Logging Setup**: Initializes logging to capture detailed execution steps and errors.
2. **User Query Collection**: Prompts the user for input and validates it.
3. **Embedding Generation**: Converts the user's query into a vector representation using OpenAI embeddings.
4. **Similarity Search**: Uses Pinecone to search for the most relevant documents based on the query embedding.
5. **Reflection and Evaluation**: Evaluates the relevance of retrieved documents and assigns scores.
6. **Query Transformation**: Refines the user's query to improve retrieval results, if needed.
7. **Context Refinement**: Combines relevant and retrieved documents to build a refined context.
8. **Response Generation**: Utilizes the refined context to generate a response using OpenAI's LLM.
9. **Self-Reflection**: Evaluates the generated response for accuracy and alignment with the query.
10. **Final Response or Query Update**: Displays the validated response or suggests an updated query for further processing.

Key Components:
---------------
- **Pinecone Integration**: For vector-based similarity search.
- **OpenAI Integration**: For embedding generation and natural language processing tasks.
- **LangGraph**: To define and manage the workflow graph.
- **State Management**: Maintains a persistent state to ensure smooth transitions across workflow nodes.

Execution Flow:
---------------
- The main function sets up the environment, defines the graph, and enters an iterative loop to
  process user queries.
- Each iteration progresses through the workflow graph, invoking the necessary nodes and updating
  the state.
- The script terminates gracefully when the user enters "exit" or encounters a critical error.
"""

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

# Constants
PINECONE_API_KEY = "9fe3c832-5827-45f6-b066-e740b9b13e33"
PINECONE_ENVIRONMENT = "us-east-1"
TEST_INDEX1 = "testindex1"
API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Global variable for embedding

global_embedding = None
global_retrieved_documents = None
# Global variable for relevance scores
global_relevance_scores = None
# Global variable for the transformed query
global_transformed_query = None
# Global variables
global_validated_answer = None
global_updated_query = None
# Global variable for relevant documents
#global_relevant_documents = None
# Global variable for relevant documents
global_relevant_documents = []



def setup_logging():
    """
    Sets up logging for the application.

    Creates the log directory if it does not exist and configures the logging settings 
    to save logs in the specified file with detailed debug information.

    Raises:
        Exception: If there's an error while setting up logging.
    """
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\LANGGRAPH-1\1'
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







def user_query_node(state):
    """
    User Query Node: Collects a single user input and updates the state.

    Prompts the user for input, checks for exit conditions, and appends the input 
    to the state using a helper function.

    Args:
        state (dict): The current state of the workflow containing the message history.

    Returns:
        dict: Updated state with the user's query appended to the messages.

    Raises:
        Exception: If there's an error processing the user input.
    """
    try:
        logging.debug("USER QUERY NODE EXECUTION STARTED.")
 
        # Prompt the user for input
        user_query = input("You: ").strip()

        # Validate input
        if not user_query:
            raise ValueError("User query is empty. Please provide a valid query.")

        # Exit condition
        if user_query.lower() == "exit":
            logging.debug("Termination condition met. Exiting conversation.")
            print("Conversation ended successfully.")
            logging.debug("USER QUERY NODE EXECUTION COMPLETED.")
            exit(0)

        # Add the user query to the state using add_messages
        messages = state.setdefault("messages", [])
        state["messages"] = add_messages(messages, [HumanMessage(content=user_query)])

        # Log details
        last_message = state["messages"][-1]
        logging.debug(f"User query added to state: {last_message.content}")
        logging.debug("USER QUERY NODE EXECUTION COMPLETED.")
        return state
    except Exception as e:
        logging.error(f"Error in USER QUERY NODE: {e}")
        raise


def query_embedding_node(state):
    """
    Query Embedding Node: Generates and stores an embedding in the global variable.

    Args:
        state (dict): Workflow state containing the user query.

    Returns:
        dict: Unmodified state.
    """
    try:
        logging.debug("QUERY EMBEDDING NODE EXECUTION STARTED.")

        # Access the global embedding variable
        global global_embedding

        # Initialize OpenAI embeddings
        embeddings_model = OpenAIEmbeddings(api_key=API_KEY)

        # Get the user query from the state
        user_query = state["messages"][-1].content
        if not user_query:
            raise ValueError("User query is empty.")

        # Generate the embedding
        embedding = embeddings_model.embed_documents([user_query])[0]
        logging.debug(f"Generated embedding for query: {user_query}")

        # Store the embedding in the global variable
        global_embedding = embedding
        logging.debug("Embedding stored in global variable.")

        logging.debug("QUERY EMBEDDING NODE EXECUTION COMPLETED.")
        return state  # Return the state unchanged
    except Exception as e:
        logging.error(f"Error in QUERY EMBEDDING NODE: {e}")
        raise









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

def similarity_search_node(state):
    """
    Similarity Search Node: Performs similarity search and stores results in a global variable.

    Args:
        state (dict): Workflow state.

    Returns:
        dict: Unmodified state.
    """
    try:
        logging.debug("SIMILARITY SEARCH NODE EXECUTION STARTED.")

        # Access global variables
        global global_embedding
        global global_retrieved_documents

        if global_embedding is None:
            raise ValueError("Global embedding is not set. Ensure query_embedding_node runs first.")

        # Retrieve the Pinecone index
        pinecone_index = initialize_pinecone()

        # Perform similarity search
        logging.debug(f"Using global embedding for similarity search: {global_embedding[:10]}... (truncated)")
        results = pinecone_index.query(vector=global_embedding, top_k=5, include_metadata=True)
        logging.debug(f"Similarity search results: {results}")

        # Extract the top results and store them in the global variable
        global_retrieved_documents = [
            {"id": match["id"], "score": match["score"], "metadata": match.get("metadata", {})}
            for match in results.get("matches", [])
        ]
        logging.debug("Retrieved documents stored in global variable.")

        logging.debug("SIMILARITY SEARCH NODE EXECUTION COMPLETED.")
        return state  # Return the state unchanged
    except Exception as e:
        logging.error(f"Error in SIMILARITY SEARCH NODE: {e}")
        raise






def reflection_node(state):
    """
    Reflection Node: Evaluates the relevance of retrieved documents to the user query.

    Uses the LLM to grade the relevance of the retrieved documents and stores the relevance
    scores in a global variable.

    Args:
        state (dict): Workflow state, including the user query.

    Returns:
        dict: Unmodified state.

    Raises:
        Exception: If the relevance evaluation fails or required global variables are missing.
    """
    try:
        logging.debug("REFLECTION NODE EXECUTION STARTED.")

        # Access global variables
        global global_retrieved_documents
        global global_relevance_scores

        # Validate state and global retrieved documents
        if "messages" not in state or not state["messages"]:
            raise ValueError("User query not found in the state.")
        if global_retrieved_documents is None:
            raise ValueError("Global retrieved documents are not set. Ensure similarity_search_node runs first.")

        # Extract the user query and retrieved documents
        user_query = state["messages"][-1].content  # Access the content attribute directly
        retrieved_documents = global_retrieved_documents

        # Prepare the system and human messages
        system_prompt = (
            "You are an evaluator. Given the user's query and the retrieved documents, "
            "assign a relevance score (0-10) to each document and provide a brief explanation."
        )
        documents_text = "\n".join(
            [f"Document {i+1}: {doc['metadata'].get('content', 'No content available')}" for i, doc in enumerate(retrieved_documents)]
        )
        human_prompt = f"User Query: {user_query}\n\nRetrieved Documents:\n{documents_text}"

        # Create prompt templates
        system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
        chat_prompt_template = ChatPromptTemplate.from_messages([system_message_template, human_message_template])

        # Format the prompt
        formatted_prompt = chat_prompt_template.format_prompt().to_messages()

        # Initialize the LLM
        llm = ChatOpenAI(api_key="sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ")

        # Invoke the LLM
        response = llm.invoke(formatted_prompt)
        logging.debug(f"LLM response: {response.content}")

        # Store relevance scores in the global variable
        global_relevance_scores = response.content
        logging.debug(f"Relevance scores stored in global variable: {global_relevance_scores}")

        logging.debug("REFLECTION NODE EXECUTION COMPLETED.")
        return state  # Return the state unchanged
    except Exception as e:
        logging.error(f"Error in REFLECTION NODE: {e}")
        raise


def query_transformation_node(state):
    """
    Query Transformation Node: Modifies the original query to improve retrieval results.

    Uses the LLM to generate a refined query based on global variables for the user query
    and relevance scores. Stores the transformed query in a global variable.

    Args:
        state (dict): Workflow state.

    Returns:
        dict: Unmodified state.

    Raises:
        Exception: If the query transformation fails or required global variables are missing.
    """
    try:
        logging.debug("QUERY TRANSFORMATION NODE EXECUTION STARTED.")

        # Access global variables
        global global_relevance_scores
        global global_transformed_query

        # Validate state for user query and global variables
        if "messages" not in state or not state["messages"]:
            raise ValueError("User query not found in the state.")
        if global_relevance_scores is None:
            raise ValueError("Global relevance scores are not set. Ensure reflection_node runs first.")

        # Extract the user query and reflection feedback
        user_query = state["messages"][-1].content
        relevance_scores = global_relevance_scores

        # Prepare the system and human prompts
        system_prompt = (
            "You are a query optimizer. Based on the user's query and the feedback "
            "from the relevance evaluation, rewrite the query to improve retrieval results."
        )
        human_prompt = (
            f"Original Query: {user_query}\n\n"
            f"Reflection Feedback:\n{relevance_scores}\n\n"
            f"Please provide a refined query suitable for better retrieval."
        )

        # Create prompt templates
        system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
        chat_prompt_template = ChatPromptTemplate.from_messages([system_message_template, human_message_template])

        # Format the prompt
        formatted_prompt = chat_prompt_template.format_prompt().to_messages()

        # Initialize the LLM
        llm = ChatOpenAI(api_key="sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ")

        # Invoke the LLM
        response = llm.invoke(formatted_prompt)
        logging.debug(f"LLM response: {response.content}")

        # Store the transformed query in the global variable
        global_transformed_query = response.content.strip()
        logging.debug(f"Transformed query stored in global variable: {global_transformed_query}")

        logging.debug("QUERY TRANSFORMATION NODE EXECUTION COMPLETED.")
        return state  # Return the state unchanged
    except Exception as e:
        logging.error(f"Error in QUERY TRANSFORMATION NODE: {e}")
        raise


def context_refinement_node(state):
    try:
        logging.debug("CONTEXT REFINEMENT NODE EXECUTION STARTED.")

        # Access global variables
        global global_retrieved_documents
        global global_relevant_documents
        global global_refined_context

        # Validate global variables
        if not global_retrieved_documents:
            raise ValueError("Global retrieved documents are not set. Ensure similarity_search_node runs first.")
        if not global_relevant_documents:
            logging.warning("Global relevant documents are empty. Proceeding with retrieved documents only.")
            global_relevant_documents = []

        # Combine and format documents
        combined_documents = global_relevant_documents + global_retrieved_documents
        consolidated_context = "\n".join([
            f"Document {i+1}: {doc.get('metadata', {}).get('content', 'No content available')}" 
            for i, doc in enumerate(combined_documents)
        ])

        # Store the refined context in the global variable
        global_refined_context = consolidated_context
        logging.debug(f"Refined context stored in global variable: {global_refined_context[:100]}... (truncated)")

        logging.debug("CONTEXT REFINEMENT NODE EXECUTION COMPLETED.")
        return state  # Return the state unchanged
    except Exception as e:
        logging.error(f"Error in CONTEXT REFINEMENT NODE: {e}")
        raise

def generation_node(state):
    """
    Generation Node: Uses the refined context and user query to generate an initial response.

    Combines the user query and refined context, sends them to the LLM, and stores the generated
    answer in a global variable.

    Args:
        state (dict): Workflow state.

    Returns:
        dict: Unmodified state.

    Raises:
        Exception: If the response generation fails or required global variables are missing.
    """
    try:
        logging.debug("GENERATION NODE EXECUTION STARTED.")

        # Access global variables
        global global_refined_context
        global global_generated_answer

        # Validate state and global variables
        if "messages" not in state or not state["messages"]:
            raise ValueError("User query not found in the state.")
        if global_refined_context is None:
            raise ValueError("Global refined context is not set. Ensure context_refinement_node runs first.")

        # Extract the user query and refined context
        user_query = state["messages"][-1].content
        refined_context = global_refined_context

        # Prepare the system and human prompts
        system_prompt = (
            "You are an intelligent assistant. Based on the provided refined context and user query, "
            "generate a helpful and accurate response."
        )
        human_prompt = f"User Query: {user_query}\n\nRefined Context:\n{refined_context}"

        # Create prompt templates
        system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
        chat_prompt_template = ChatPromptTemplate.from_messages([system_message_template, human_message_template])

        # Format the prompt
        formatted_prompt = chat_prompt_template.format_prompt().to_messages()

        # Initialize the LLM
        llm = ChatOpenAI(api_key="sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ")

        # Invoke the LLM
        response = llm.invoke(formatted_prompt)
        logging.debug(f"LLM response: {response.content}")

        # Store the generated answer in the global variable
        global_generated_answer = response.content.strip()
        logging.debug(f"Generated answer stored in global variable: {global_generated_answer}")

        logging.debug("GENERATION NODE EXECUTION COMPLETED.")
        return state  # Return the state unchanged
    except Exception as e:
        logging.error(f"Error in GENERATION NODE: {e}")
        raise


def self_reflection_node(state):
    """
    Self-Reflection Node: Evaluates the quality and factuality of the generated answer.

    Uses the LLM to assess alignment and factuality, and stores the result (validated answer
    or updated query) in global variables.

    Args:
        state (dict): Workflow state.

    Returns:
        dict: Unmodified state.

    Raises:
        Exception: If the self-reflection process fails or required global variables are missing.
    """
    try:
        logging.debug("SELF-REFLECTION NODE EXECUTION STARTED.")

        # Access global variables
        global global_generated_answer
        global global_validated_answer
        global global_updated_query

        # Validate state and global variables
        if "messages" not in state or not state["messages"]:
            raise ValueError("User query not found in the state.")
        if global_generated_answer is None:
            raise ValueError("Global generated answer is not set. Ensure generation_node runs first.")

        # Extract the user query and generated answer
        user_query = state["messages"][-1].content
        generated_answer = global_generated_answer

        # Prepare the system and human prompts
        system_prompt = (
            "You are an evaluator. Given the user's query and the generated answer, "
            "assess whether the answer is accurate and aligns with the query. "
            "If the answer is correct, confirm its accuracy. If not, suggest how to improve it "
            "or propose a better query for re-retrieval."
        )
        human_prompt = f"User Query: {user_query}\n\nGenerated Answer: {generated_answer}"

        # Create prompt templates
        system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
        chat_prompt_template = ChatPromptTemplate.from_messages([system_message_template, human_message_template])

        # Format the prompt
        formatted_prompt = chat_prompt_template.format_prompt().to_messages()

        # Initialize the LLM
        llm = ChatOpenAI(api_key="sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ")

        # Invoke the LLM
        response = llm.invoke(formatted_prompt)
        logging.debug(f"LLM response: {response.content}")

        # Parse the LLM response
        reflection_output = response.content.strip()
        if "improve" in reflection_output.lower() or "suggest" in reflection_output.lower():
            global_updated_query = reflection_output
            logging.debug(f"Updated query stored in global variable: {global_updated_query}")
        else:
            global_validated_answer = reflection_output
            logging.debug(f"Validated answer stored in global variable: {global_validated_answer}")

        logging.debug("SELF-REFLECTION NODE EXECUTION COMPLETED.")
        return state  # Return the state unchanged
    except Exception as e:
        logging.error(f"Error in SELF-REFLECTION NODE: {e}")
        raise


def end_node(state):
    """
    End Node: Returns the final, validated response to the user.

    Displays the validated answer or a message indicating the need for re-retrieval
    if an updated query was suggested, using global variables.

    Args:
        state (dict): Workflow state (unused here).

    Returns:
        None: Outputs the final response directly to the user.

    Raises:
        Exception: If the global variables do not contain the necessary information for display.
    """
    try:
        logging.debug("END NODE EXECUTION STARTED.")

        # Access global variables
        global global_validated_answer
        global global_updated_query

        # Check for a validated answer
        if global_validated_answer:
            print("\nFinal Response:")
            print(global_validated_answer)
            logging.debug(f"Final response displayed to the user: {global_validated_answer}")
        elif global_updated_query:
            # If the workflow requires re-retrieval, notify the user
            print("\nWorkflow Suggestion:")
            print("The generated response could not be fully validated.")
            print(f"Suggested Updated Query: {global_updated_query}")
            logging.debug(f"Updated query suggested to the user: {global_updated_query}")
        else:
            raise ValueError("No validated answer or updated query found in global variables.")

        logging.debug("END NODE EXECUTION COMPLETED.")
    except Exception as e:
        logging.error(f"Error in END NODE: {e}")
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
    builder.add_node("user_query", user_query_node)
    builder.add_node("query_embedding", query_embedding_node)
    builder.add_node("similarity_search", similarity_search_node)
    builder.add_node("reflection", reflection_node)
    builder.add_node("query_transformation", query_transformation_node)
    builder.add_node("context_refinement", context_refinement_node)
    builder.add_node("generation", generation_node)
    builder.add_node("self_reflection", self_reflection_node)
    builder.add_node("end", end_node)

    # Add edges to define the workflow
    builder.add_edge(START, "user_query")
    builder.add_edge("user_query", "query_embedding")
    builder.add_edge("query_embedding", "similarity_search")
    builder.add_edge("similarity_search", "reflection")
    builder.add_edge("reflection", "query_transformation")
    builder.add_edge("query_transformation", "context_refinement")
    builder.add_edge("context_refinement", "generation")
    builder.add_edge("generation", "self_reflection")
    builder.add_edge("self_reflection", "end")


    # Initialize Memory Saver
    memory_saver = MemorySaver()
    logging.debug("Memory saver initialized.")

    # Compile graph with interrupt handling
    graph = builder.compile(checkpointer=memory_saver)
    logging.debug("Workflow graph compiled successfully.")
    return graph


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

def main():
    """
    Main function to execute the workflow graph.
    """
    try:
        # Setup logging
        setup_logging()

        # Initialize LLM and Pinecone and embeddings model

        pinecone_index = initialize_pinecone()
        embeddings_model = OpenAIEmbeddings(api_key=API_KEY)

        # Define and compile the graph
        graph = define_graph()

        # Initialize state
        initial_state = {"messages": [], "retrieved_documents": [], "relevant_documents": []}
        iteration = 0

        # Run the workflow
        while True:
            iteration += 1
            logging.debug(f"START ITERATION - {iteration}")

            # Invoke the graph
            updated_state = graph.invoke(initial_state, config={"pinecone_index": pinecone_index, "embeddings_model": embeddings_model, "thread_id": "1"})
             

            # Log messages state to file
            write_all_messages_to_file(updated_state)

            # Update state for the next iteration
            initial_state = updated_state
            write_all_message_types_to_txt(updated_state.values())
            logging.debug(f"END ITERATION - {iteration}")

    except Exception as e:
        logging.error(f"Error during workflow execution: {e}")
        raise


if __name__ == "__main__":
    main()
