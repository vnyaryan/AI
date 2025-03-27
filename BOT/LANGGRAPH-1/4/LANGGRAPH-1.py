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
from pydantic import BaseModel
from langchain_core.messages import AnyMessage
from typing import List
import sys

# Constants


OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"
PINECONE_API_KEY = "9fe3c832-5827-45f6-b066-e740b9b13e33"
PINECONE_ENVIRONMENT = "us-east-1"
TEST_INDEX1 = "testindex1"

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)



class OverallState(BaseModel):
    exit_status: bool = False
    messages: List[AnyMessage] = []
    user_embedding: List[float] = []
    retrieved_documents: str = ""  
    wants_more_azure_cli: bool = False  # tracks the user’s yes/no response

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
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\LANGGRAPH-1\4'
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


def write_all_messages_to_file(state_values, output_file="msg_details.txt"):
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
                {"messages": [message.__dict__ for message in state_values]},
                indent=4
            ))
            file.write("\n======== END OF ITERATION ========\n\n")
        logging.debug(f"Messages state successfully logged to {output_file}.")
    except Exception as e:
        logging.error(f"Error writing messages to file: {e}")
        raise

# Function to write MessagesState details to a text file
def write_all_message_types_to_txt(messages, output_file="all_msg.txt"):
    """
    Processes and writes the content of all message types (HumanMessage, AIMessage, ToolMessage, etc.)
    to a text file with separators.
    
    Args:
        messages (list): A list containing message objects (e.g., HumanMessage, AIMessage, etc.).
        output_file (str): Path to the file where details will be logged. Defaults to 'all_msg.txt'.
    """
    try:
        with open(output_file, "w") as file:
            file.write("======== START OF MESSAGES ========\n\n")
            
            for message in messages:
                # Process HumanMessage
                if isinstance(message, HumanMessage):
                    file.write(f"HumanMessage Content: {message.content}\n")
                    file.write("-------------------------------------------------------------------\n")
                
                # Process AIMessage
                elif isinstance(message, AIMessage):
                    file.write(f"AIMessage Content: {message.content}\n")
                    file.write("-------------------------------------------------------------------\n")
                
                # Process SystemMessage
                elif isinstance(message, SystemMessage):
                    file.write(f"SystemMessage Content: {message.content}\n")
                    file.write("-------------------------------------------------------------------\n")
                
                # Process ToolMessage
                elif isinstance(message, ToolMessage):
                    file.write(f"ToolMessage Content: {message.content}\n")
                    file.write(f"  Tool Name: {message.name}\n")
                    file.write("-------------------------------------------------------------------\n")
                
                # Handle unknown message types
                else:
                    file.write(f"Unknown Message Type: {type(message).__name__}\n")
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


def node1(state: OverallState):
    """
    Node 1: Collects a single user input and updates the state if termination is requested.
    """
    try:
        logging.debug("NODE1 EXECUTION STARTED.")
        
        # Prompt the user for input
        user_message = input("You: ")

        # Check for termination condition
        if user_message.lower() == "exit":
            logging.debug("User requested termination in Node 1.")
            state.exit_status = True
            print("Goodbye!")
            sys.exit(0)
            

        # Process the user input (additional logic can go here)
        logging.debug(f"User message: {user_message}")
        state.messages = add_messages(state.messages, [HumanMessage(content=user_message)])
        logging.debug(f"User message added to state: {state.messages[-1].content}")
        logging.debug("NODE1 EXECUTION COMPLETED.")
        return state
    except Exception as e:
        logging.error("Error in node_1: %s", exc_info=True)
        raise



def node2(state: OverallState):
    """
    Node 2: Generates and stores an embedding for the latest HumanMessage in the OverallState.

    Args:
        state (OverallState): The current workflow state.

    Returns:
        OverallState: Updated state with the embedding of the latest HumanMessage stored in 'user_embedding'.

    Raises:
        ValueError: If no valid HumanMessage is found in the state's messages.
    """
    try:
        logging.debug("NODE2 EXECUTION STARTED.")

        # Check if 'messages' exists and is not empty
        if not state.messages:
            raise ValueError("The 'messages' list in OverallState is empty.")

        # Find the latest HumanMessage
        latest_human_message = None
        for message in reversed(state.messages):
            if isinstance(message, HumanMessage):
                latest_human_message = message
                break

        if not latest_human_message:
            raise ValueError("No valid HumanMessage found in OverallState.messages.")

        # Concatenate all human and AI messages for embedding
        full_history = "\n".join([
            f"{msg.content}" for msg in state.messages if isinstance(msg, (HumanMessage, AIMessage))
        ])

        # Extract the content of the latest HumanMessage
        user_query = latest_human_message.content
        if not user_query:
            raise ValueError("User query in HumanMessage is empty.")

        logging.debug(f"Latest HumanMessage content: {user_query}")

        # Generate the embedding
        embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        #embedding = embeddings_model.embed_documents([user_query])[0]
        embedding = embeddings_model.embed_documents([full_history])[0]
        logging.debug(f"Generated embedding for query '{user_query}': {embedding}")

        # Store the embedding in the 'user_embedding' attribute of OverallState
        state.user_embedding = embedding
        logging.debug("Embedding stored in OverallState.user_embedding.")

        logging.debug("NODE2 EXECUTION COMPLETED.")
        return state
    except Exception as e:
        logging.error(f"Error in NODE2: {e}")
        raise



def node2(state: OverallState):
    try:
        # Concatenate all human and AI messages for embedding
        full_history = "\n".join([
            f"{msg.content}" for msg in state.messages if isinstance(msg, (HumanMessage, AIMessage))
        ])
        
        if not full_history:
            raise ValueError("The message history is empty.")
        
        # Generate embedding for the concatenated history
        embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        embedding = embeddings_model.embed_documents([full_history])[0]

        # Store the embedding in state
        state.user_embedding = embedding
        return state
    except Exception as e:
        logging.error(f"Error in NODE2 (full-history embedding): {e}")
        raise


def node3(state: OverallState):
    """
    Similarity Search Node: Performs similarity search using the user embedding and stores results
    in the 'retrieved_documents' attribute of OverallState.

    Args:
        state (OverallState): The current workflow state.

    Returns:
        OverallState: Updated state with retrieved similarity search results stored in 'retrieved_documents'.

    Raises:
        ValueError: If user embedding is not set or an error occurs during the similarity search.
    """
    try:
        logging.debug("NODE3 EXECUTION STARTED.")

        # Ensure the embedding is available in the state
        if not state.user_embedding:
            raise ValueError("User embedding is not set. Ensure node3 (query_embedding_node) runs first.")

        # Retrieve the Pinecone index
        pinecone_index = initialize_pinecone()

        # Perform similarity search
        logging.debug(f"Using user embedding for similarity search: {state.user_embedding[:10]}... (truncated)")
        results = pinecone_index.query(vector=state.user_embedding, top_k=3, include_metadata=True)

        logging.debug(f"Similarity search results: {results}")

        # Sort results by score
        sorted_results = sorted(results['matches'], key=lambda x: x['score'])
        logging.debug(f"Sorted similarity search results: {sorted_results}")

        # Generate additional context from the top 3 results using `values`
        additional_context = "\n".join([str(result['values']) for result in sorted_results[:3]])
        logging.debug(f"Additional context extracted from top 3 results: {additional_context}")

        # Extract and store the top results in the state's 'retrieved_documents' attribute
        # state.retrieved_documents = [
        #     {"id": match["id"], "score": match["score"], "metadata": match.get("metadata", {}), "values": match.get("values", "")}
        #     for match in sorted_results
        # ]
        # logging.debug(f"Retrieved documents stored in state.retrieved_documents: {state.retrieved_documents}")

        # Save the additional context directly in 'retrieved_documents'
        state.retrieved_documents = additional_context
        logging.debug(f"'retrieved_documents' updated with additional context.")


        logging.debug("NODE3 EXECUTION COMPLETED.")
        return state
    except Exception as e:
        logging.error(f"Error in SIMILARITY SEARCH NODE3: {e}")
        raise


def node4(state: OverallState):
    """
    Generates a response for the user using `additional_context` stored in `retrieved_documents`
    and the latest user query. Truncates context directly within the code, formats prompts,
    and saves the input/output context to memory.

    Args:
        state (OverallState): The current workflow state.
 

    Returns:
        str: The AI's response generated using the user query and additional context.

    Raises:
        ValueError: If `retrieved_documents` or the user query is missing.
    """
    try:
        logging.debug("NODE4 EXECUTION STARTED.")

        # Check if additional context is available in `retrieved_documents`
        additional_context = state.retrieved_documents
        if not additional_context:
            raise ValueError("No additional context found in 'retrieved_documents'.")

        # Extract the latest user query from messages
        latest_human_message = next(
            (message for message in reversed(state.messages) if isinstance(message, HumanMessage)),
            None
        )
        if not latest_human_message:
            raise ValueError("No valid HumanMessage found in OverallState.messages.")

        user_query = latest_human_message.content
        if not user_query:
            raise ValueError("User query in HumanMessage is empty.")

        logging.debug(f"User query: {user_query}")
        logging.debug(f"Additional context: {additional_context}")

        # Truncate the context directly to fit within token limits
        max_chars = 2000
        truncated_context = additional_context[:max_chars]  # Direct truncation logic
        logging.debug(f"Truncated context: {truncated_context}")

        # Define templates
        system_template = (
            "You are an expert in Azure and az CLI commands. "
            "Use the provided context to generate an appropriate Azure CLI command for the user's task."
        )
        human_template = "User task: {user_query}\nContext: {truncated_context}"

        # Create prompt templates
        system_prompt_template = SystemMessagePromptTemplate.from_template(system_template)
        human_prompt_template = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt_template = ChatPromptTemplate.from_messages([system_prompt_template, human_prompt_template])

        # Format the prompt and serialize messages
        formatted_prompt = chat_prompt_template.format_prompt(
            user_query=user_query,  # Fixed key name
            truncated_context=truncated_context
        )
        messages = formatted_prompt.to_messages()
        serialized_messages = [
            {"role": "system", "content": msg.content} if isinstance(msg, SystemMessage)
            else {"role": "user", "content": msg.content}
            for msg in messages
        ]

        # Send the prompt to the LLM
        response = llm.invoke(serialized_messages)
        logging.debug(f"LLM response: {response.content}")

        # Append the AI's response to the state's messages
        state.messages = add_messages(state.messages, [response])
        logging.debug("AI response added to OverallState.messages.")


        # **Print the AI response right here!**
        print(response.content)

        logging.debug("NODE4 EXECUTION COMPLETED.")
        return state
    except Exception as e:
        logging.error(f"Error in GENERATE RESPONSE WITH DOCUMENTS NODE4: {e}")
        raise

def node5(state: OverallState):
    """
    This node asks the user if they want more Azure CLI details after providing an AI response.
    """
    try:
        logging.debug("NODE5_ASK_FOLLOWUP EXECUTION STARTED.")


        followup_text = (
            "Would you like more details about this Azure CLI command? "
            "You can say 'yes' for more info or 'no' to move on."
        )
        ai_followup_message = AIMessage(content=followup_text)

        state.messages = add_messages(state.messages, [ai_followup_message])
        
        logging.debug(f"Follow-up question to user: {ai_followup_message.content}")
        logging.debug("NODE5_ASK_FOLLOWUP EXECUTION COMPLETED.")

        # **Print the AI response right here!**
        print(ai_followup_message.content)
        return state

    except Exception as e:
        logging.error("Error in node5_ask_followup: %s", e)
        raise



def node6(state: OverallState):
    """
    Captures the user's response to whether they want more Azure CLI details or not.
    """
    try:
        logging.debug("NODE6_CAPTURE_FOLLOWUP_RESPONSE EXECUTION STARTED.")
        
        user_message = input("Yes/No: ")
        logging.debug(f"User follow-up response: {user_message}")

        state.messages = add_messages(state.messages, [HumanMessage(content=user_message)])
        
        # Simple logic: check if user said "yes" or "no"
        if user_message.strip() in ["Yes", "yes", "y"]:
            state.wants_more_azure_cli = True
        else:
            state.wants_more_azure_cli = False
            # Here is where you can add a final console message or some cleanup:
            print("No more details? Okay, have a great day!")
            sys.exit(0)


        logging.debug(f"wants_more_azure_cli: {state.wants_more_azure_cli}")
        logging.debug("NODE6_CAPTURE_FOLLOWUP_RESPONSE EXECUTION COMPLETED.")
        return state
    except Exception as e:
        logging.error("Error in node6_capture_followup_response: %s", e)
        raise


def define_graph():
    logging.debug("Defining the workflow graph.")
    
    builder = StateGraph(OverallState)

    # 1) Add your nodes
    builder.add_node("user_query", node1)         # Asks user for next question
    builder.add_node("generate_embedding", node2) # Embeds user input
    builder.add_node("similarity_search", node3)  # Queries Pinecone
    builder.add_node("ai_response", node4)        # Generates AI response
    builder.add_node("ask_followup", node5)       # AI asks: "Do you want more info?"
    builder.add_node("capture_followup", node6)   # User responds yes/no

    # 2) Define standard edges
    builder.add_edge(START, "user_query")
    builder.add_edge("user_query", "generate_embedding")
    builder.add_edge("generate_embedding", "similarity_search")
    builder.add_edge("similarity_search", "ai_response")
    builder.add_edge("ai_response", "ask_followup")
    builder.add_edge("ask_followup", "capture_followup")

    # 3) Define a helper function for the conditional routing
    def route_followup(state: OverallState) -> str:
        """
        Decide where to go next based on user's response in capture_followup.
        """
        logging.debug("Entering route_followup function.")

        if state.wants_more_azure_cli:
            logging.debug("User wants more Azure CLI details. Routing back to 'user_query'.")
            return "user_query"
        else:
            logging.debug("User declined more Azure CLI details. Exiting conversation.")
            print("Exiting conversation.")
            return END




    # 4) Use add_conditional_edges for branching
    builder.add_conditional_edges("capture_followup", route_followup)

    # 5) Initialize memory saver and compile the graph
    memory_saver = MemorySaver()
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
        
        # Initialize the state
        initial_state = OverallState(exit_status=False,messages=[SystemMessage(content=system_prompt)])


        # Define the graph
        graph = define_graph()

        # Iteratively invoke the graph until termination
        iteration = 0
        while True:  # Infinite loop until explicit termination
            iteration += 1
            logging.debug(f"START ITERATION - {iteration}")

            # Invoke the graph
            updated_state = graph.invoke(initial_state, config={"configurable": {"thread_id": 1}})
            logging.debug(f"Updated State: {updated_state}")
            logging.debug(f"Type of updated_state: {type(updated_state)}")
            logging.debug(f"Updated State content: {updated_state}")

            # Print the AI's response if it's the latest message
            if isinstance(updated_state["messages"][-1], AIMessage):
                print(f"AI: {updated_state['messages'][-1].content}")

            # Write messages to a static file `all_msg.txt`
            write_all_messages_to_file(updated_state["messages"])
            write_all_message_types_to_txt(updated_state["messages"])


            # Update the initial statex
            # Convert updated_state to OverallState
            initial_state = OverallState(**dict(updated_state))



            logging.debug(f"ENDING ITERATION - {iteration}")

    except Exception as e:
        logging.error(f"Error during multi-turn conversation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
