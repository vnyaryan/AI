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
    next_node: str = ""  
    assessment_status: Literal["GOOD", "BAD"] = "GOOD" 


 



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
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\LANGGRAPH-1\5'
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
        with open(output_file, "w") as file:
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

# callable function for conditional edges
def next_node_selector(state):
    if state.next_node == "generate_user_query_embedding":
        return "generate_embedding"
    elif state.next_node == "ai_response":
        return "ai_response"
    else:
        raise ValueError(f"Invalid next_node value: {state.next_node}")

def relevance_assessment_selector(state: OverallState):
    """
    Determines the next node based on the `assessment_status` attribute.
    If `assessment_status` is "GOOD", the workflow ends.
    If `assessment_status` is "BAD", the workflow moves to the clarification node.
    """
    if state.assessment_status == "GOOD":
        return END
    elif state.assessment_status == "BAD":
        return "clarification_node"
    else:
        raise ValueError(f"Invalid assessment_status: {state.assessment_status}")


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
    Node 2: Evaluates the message history to determine if similarity search is required.
    If required, refines the user query and prepares it for embedding generation.

    Args:
        state (OverallState): The current workflow state.

    Returns:
        OverallState: Updated state with decision and refined query.

    Raises:
        ValueError: If message history is missing or the LLM response is invalid.
    """
    try:
        logging.debug("NODE2: EVALUATE_AND_REFINE_QUERY STARTED.")

        # Combine all messages into a single input for the LLM
        full_message_history = "\n".join(
            f"{type(msg).__name__}: {msg.content}" for msg in state.messages
        )
        if not full_message_history:
            raise ValueError("The message history is empty.")

        logging.debug(f"Full message history:\n{full_message_history}")

        # Step 1: Ask the LLM if similarity search is required
        evaluation_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an intelligent assistant. Analyze the following conversation history "
                "and determine if latest human message related azure cli commands or not"
                "Respond with 'Yes' if it's related to azure az cli command, or 'No' if not related to azure az cli."
            ),
            HumanMessagePromptTemplate.from_template("Conversation History:\n{history}")
        ]).format_prompt(history=full_message_history).to_messages()

        response = llm.invoke(evaluation_prompt)
        llm_decision = response.content.strip().lower()
        logging.debug(f"LLM evaluation result: {llm_decision}")

        # Step 2: Refine the user query if similarity search is required
        if llm_decision == "yes":
            logging.debug("LLM decided similarity search is required. Refining user query.")

            # Create a prompt for refining the user query
            refinement_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are an intelligent assistant. Based on the conversation history, "
                    "refine the user's query to make it more specific and suitable for a similarity search."
                ),
                HumanMessagePromptTemplate.from_template("Conversation History:\n{history}")
            ]).format_prompt(history=full_message_history).to_messages()

            refinement_response = llm.invoke(refinement_prompt)
            refined_query = refinement_response.content.strip()
            if not refined_query:
                raise ValueError("Refined query is empty.")
            logging.debug(f"Refined user query: {refined_query}")

            # Update the state with the refined query
            state.retrieved_documents = "Initiating similarity search..."
            state.messages = add_messages(state.messages, [HumanMessage(content=refined_query)])
            state.next_node = "generate_user_query_embedding"
        elif llm_decision == "no":
            logging.debug("LLM decided similarity search is not required.")
            state.retrieved_documents = ""  # No similarity search context
            state.next_node = "ai_response"
        else:
            raise ValueError(f"Unexpected LLM response: {llm_decision}")

        logging.debug("NODE2: EVALUATE_AND_REFINE_QUERY COMPLETED.")
        return state
    except Exception as e:
        logging.error(f"Error in NODE2: EVALUATE_AND_REFINE_QUERY: {e}")
        raise

def node3(state: OverallState):
    """
    Node 3: Generates and stores an embedding for the refined query from Node2.

    Args:
        state (OverallState): The current workflow state.

    Returns:
        OverallState: Updated state with the embedding of the refined query stored in 'user_embedding'.

    Raises:
        ValueError: If no refined query is found in the state's messages.
    """
    try:
        logging.debug("NODE3 EXECUTION STARTED.")

        # Check if similarity search is required
        if state.next_node != "generate_user_query_embedding":
            logging.debug("Similarity search not required. Skipping embedding generation.")
            return state

        # Retrieve the refined query from the last HumanMessage
        latest_human_message = next(
            (message for message in reversed(state.messages) if isinstance(message, HumanMessage)),
            None
        )

        if not latest_human_message:
            raise ValueError("No valid HumanMessage found in OverallState.messages.")

        refined_query = latest_human_message.content
        if not refined_query:
            raise ValueError("Refined query from Node2 is empty.")

        logging.debug(f"Using refined query for embedding: {refined_query}")

        # Generate the embedding
        embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        embedding = embeddings_model.embed_documents([refined_query])[0]
        logging.debug(f"Generated embedding for refined query: {embedding[:10]}... (truncated)")

        # Store the embedding in the 'user_embedding' attribute of OverallState
        state.user_embedding = embedding
        logging.debug("Embedding stored in OverallState.user_embedding.")

        logging.debug("NODE3 EXECUTION COMPLETED.")
        return state
    except Exception as e:
        logging.error(f"Error in NODE3: {e}")
        raise





def node4(state: OverallState):
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
        logging.debug("NODE4 EXECUTION STARTED.")

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


        # Save the additional context directly in 'retrieved_documents'
        state.retrieved_documents = additional_context
        logging.debug(f"'retrieved_documents' updated with additional context.")


        logging.debug("NODE4 EXECUTION COMPLETED.")
        return state
    except Exception as e:
        logging.error(f"Error in SIMILARITY SEARCH NODE4: {e}")
        raise


def node5(state: OverallState):
    """
    Node 5: Generates a response for the user based on `retrieved_documents` 
    and the entire message history. If similarity search was not performed 
    (retrieved_documents is empty), it relies solely on the message history.

    Args:
        state (OverallState): The current workflow state.

    Returns:
        OverallState: Updated state with the AI's response added to the messages.

    Raises:
        ValueError: If the message history is missing or an unexpected error occurs.
    """
    try:
        logging.debug("NODE5 EXECUTION STARTED.")

        # Combine all messages into a single input for the LLM
        full_message_history = "\n".join(
            f"{type(msg).__name__}: {msg.content}" for msg in state.messages
        )
        if not full_message_history:
            raise ValueError("The message history is empty.")

        logging.debug(f"Full message history:\n{full_message_history}")

        # Check for additional context in `retrieved_documents`
        additional_context = state.retrieved_documents or ""
        if additional_context:
            logging.debug("Additional context found from similarity search.")
        else:
            logging.debug("No additional context available. Proceeding with message history only.")

        # Define the LLM prompt
        response_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an intelligent assistant skilled in Azure and az CLI commands. "
                "Using the provided conversation history and any additional context, "
                "generate the most appropriate response to address the user's task or query."
            ),
            HumanMessagePromptTemplate.from_template(
                "Conversation History:\n{history}\n\nAdditional Context:\n{context}"
            )
        ]).format_prompt(history=full_message_history, context=additional_context).to_messages()

        # Invoke the LLM with the formatted prompt
        response = llm.invoke(response_prompt)
        if not response or not response.content.strip():
            raise ValueError("LLM returned an empty response.")

        logging.debug(f"LLM response: {response.content}")

        # Append the LLM's response to the state's messages
        state.messages = add_messages(state.messages, [response])
        logging.debug("AI response successfully added to the state.")

        # Output the AI response to the user
        #print(response.content)

        logging.debug("NODE5 EXECUTION COMPLETED.")
        return state
    except Exception as e:
        logging.error(f"Error in NODE5: {e}")
        raise

def node6(state: OverallState):
    """
    Node6: Validates the relevance of the AI response.
    If the response is good, it prints the AI response.
    If the response is bad, it transitions to Node7 (Clarification Node).
    """
    try:
        logging.debug("NODE6: RELEVANCE ASSESSMENT STARTED.")

        # Retrieve the latest AI response
        latest_ai_message = next(
            (msg for msg in reversed(state.messages) if isinstance(msg, AIMessage)),
            None
        )
        if not latest_ai_message:
            raise ValueError("No AI response found for relevance assessment.")

        # Create a prompt for relevance validation
        relevance_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an intelligent assistant. Determine if the following AI response is a good fit "
                "for the user's query based on the conversation history. Respond with 'GOOD' or 'BAD'."
            ),
            HumanMessagePromptTemplate.from_template(
                "User Question:\n{question}\n\nAI Response:\n{response}\n\nConversation History:\n{history}"
            )
        ]).format_prompt(
            question=state.messages[-2].content,  # Assuming second last message is the user's question
            response=latest_ai_message.content,
            history="\n".join([f"{type(msg).__name__}: {msg.content}" for msg in state.messages])
        ).to_messages()

        # Get the LLM's assessment
        response = llm.invoke(relevance_prompt)
        assessment = response.content.strip().upper()

        # Update assessment_status
        if assessment == "GOOD":
            logging.debug("AI response is relevant and appropriate.")
            state.assessment_status = "GOOD"
        elif assessment == "BAD":
            logging.warning("AI response is not relevant. Redirecting to clarification node.")
            state.assessment_status = "BAD"
        else:
            logging.error(f"Unexpected assessment result: {assessment}")
            state.assessment_status = "BAD"

        # Print AI response if status is GOOD
        if state.assessment_status == "GOOD":
            print(latest_ai_message.content)
            logging.debug("AI response printed successfully.")
        else:
            logging.debug("Relevance assessment marked as BAD. Moving to clarification node.")

        logging.debug("NODE6: RELEVANCE ASSESSMENT COMPLETED.")
        return state

    except Exception as e:
        logging.error(f"Error in NODE6: RELEVANCE ASSESSMENT: {e}")
        raise

def node7(state: OverallState):
    """
    Node7: Generates a clarification query when the AI response is assessed as 'BAD'.
    """
    try:
        logging.debug("NODE7: CLARIFICATION NODE STARTED.")

        # Combine history and AI response for context
        history = "\n".join([f"{type(msg).__name__}: {msg.content}" for msg in state.messages])
        latest_ai_message = next(
            (msg for msg in reversed(state.messages) if isinstance(msg, AIMessage)),
            None
        )
        if not latest_ai_message:
            raise ValueError("No AI response found for clarification.")

        # Create a prompt to generate a clarification query
        clarification_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "The user's query may need more clarity. Based on the conversation history and the AI's response, "
                "generate a clarification question to ask the user for more details."
            ),
            HumanMessagePromptTemplate.from_template(
                "AI Response:\n{response}\n\nConversation History:\n{history}"
            )
        ]).format_prompt(
            response=latest_ai_message.content,
            history=history
        ).to_messages()

        # Get the clarification question
        clarification_response = llm.invoke(clarification_prompt)
        clarification_question = clarification_response.content.strip()

        # Update the state with the clarification question
        state.messages = add_messages(state.messages, [HumanMessage(content=clarification_question)])
        logging.debug(f"Generated clarification question: {clarification_question}")

        print(clarification_question)
        logging.debug("NODE7: CLARIFICATION NODE COMPLETED.")
        return state

    except Exception as e:
        logging.error(f"Error in NODE7: CLARIFICATION NODE: {e}")
        raise

def define_graph():
    logging.debug("Defining the workflow graph.")    
    builder = StateGraph(OverallState)

    # Add Nodes
    builder.add_node("user_query", node1)                  # Asks user for input
    builder.add_node("evaluate_similarity", node2)         # Evaluates similarity search requirement
    builder.add_node("generate_embedding", node3)          # Generates embedding if needed
    builder.add_node("similarity_search", node4)           # Performs similarity search
    builder.add_node("ai_response", node5)                 # Generates AI response
    builder.add_node("relevance_assessment", node6)  # Assess AI response relevance
    builder.add_node("clarification_node", node7)            # Ask for user clarification


    # Add Edges
    builder.add_edge(START, "user_query")
    builder.add_edge("user_query", "evaluate_similarity")
    builder.add_conditional_edges("evaluate_similarity", next_node_selector)
    builder.add_edge("generate_embedding", "similarity_search")
    builder.add_edge("similarity_search", "ai_response")
    builder.add_edge("ai_response", "relevance_assessment")
    builder.add_conditional_edges("relevance_assessment", relevance_assessment_selector)

    # Compile the graph
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
            #logging.debug(f"Updated State: {updated_state}")
            #logging.debug(f"Type of updated_state: {type(updated_state)}")
            #logging.debug(f"Updated State content: {updated_state}")



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
