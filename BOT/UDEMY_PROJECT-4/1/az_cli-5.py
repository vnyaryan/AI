"""
This script interacts with the OpenAI API and pre-existing Pinecone indices to generate, refine, and validate Azure CLI commands based on user input using vector-based similarity search.

Logic of the script:
1. Accept a user query and convert it into a vector using embeddings.
2. Perform a similarity search using the generated vector.
3. Use the similarity search results as context to generate the Azure CLI command.
4. Display the refined command until the user is satisfied.

Input Parameters:
1. API_KEY: The OpenAI API key for authentication.
2. PINECONE_API_KEY: The Pinecone API key for accessing the service.
3. PINECONE_ENVIRONMENT: Environment details for Pinecone.
4. Index details: `testindex` and `testindex1`.
"""

import os
import logging
import sys
import pinecone
from pinecone import ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings

# Constants for logging and API setup
LOG_DIRECTORY = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-3\3'
LOG_FILENAME = 'az_cli-5.log'
API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"
PINECONE_API_KEY = "9fe3c832-5827-45f6-b066-e740b9b13e33"
PINECONE_ENVIRONMENT = "us-east-1"
TEST_INDEX1 = "testindex1"
TEST_INDEX = "testindex"

# Set up logging configuration to track script execution
def setup_logging():
    try:
        # Ensure the log directory exists
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(LOG_DIRECTORY, LOG_FILENAME),
            filemode='a'  # Append mode
        )
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        sys.exit(1)

# Terminate the script with an optional error message
def terminate_script(exit_code=1, message=None):
    if message:
        logging.error(message)
    sys.exit(exit_code)

# Perform similarity search to get additional context for the user's query
def perform_similarity_search(user_task, embeddings_model, index):
    logging.debug("Generating embedding for similarity search.")
    embedding = embeddings_model.embed_documents([user_task])[0]
    try:
        # Perform similarity search on the specified index
        results = index.query(vector=embedding, top_k=3, include_values=True)

        # Sort results based on similarity scores
        results = sorted(results['matches'], key=lambda x: x['score'])

        # Extract the actual content from the top results and ensure it's a string
        additional_context = "\n".join([str(result['values']) for result in results[:3]])
        logging.debug("Similarity search results obtained successfully.")
        return additional_context
    except Exception as e:
        logging.error(f"Failed to perform similarity search: {e}")
        raise

# Truncate the context to a specified number of characters before using it in the prompt
def truncate_context(context, max_chars=2000):
    """Truncate the context to the specified maximum number of characters."""
    return context[:max_chars]

# Generate an Azure CLI command using the context from similarity search
def generate_azure_cli_command_with_context(user_task, llm, context):
    logging.debug("Generating Azure CLI command using context from similarity search.")
    truncated_context = truncate_context(context, max_chars=2000)  # Adjust character limit as needed.
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are an expert of Azure and az cli. Use the following context to generate an appropriate Azure az cli command as per the user’s task."
            ),
            HumanMessagePromptTemplate.from_template(
                "User task: {user_task}\nContext: {context}"
            )
        ]
    )
    azure_cli_request_prompt = chat_prompt_template.format_prompt(
        user_task=user_task,
        context=truncated_context
    ).to_messages()
    
    response = llm.invoke(azure_cli_request_prompt)
    return response.content

# Get feedback from the user to refine the command
def get_user_feedback():
    return input("\nIs the generated Azure CLI command correct? (yes/no): ").strip().lower()

# Initialize Pinecone Index
def initialize_pinecone(index_name):
    """Initialize the Pinecone index."""
    try:
        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        # Retrieve the index
        index = pc.Index(index_name)
        logging.debug(f"Pinecone index '{index_name}' initialized.")
        print(f"Pinecone index '{index_name}' initialized.")
        return index
    except Exception as e:
        logging.error(f"Failed to initialize Pinecone: {e}")
        raise

# Main function to drive the interaction with the user
def main():
    setup_logging()

    # Initialize the LLM
    llm = ChatOpenAI(api_key=API_KEY)
    logging.debug("ChatOpenAI initialized with provided API key.")

    # Initialize Pinecone and embeddings
    embeddings_model = OpenAIEmbeddings(api_key=API_KEY)
    index_test1 = initialize_pinecone(TEST_INDEX1)
    index_test = initialize_pinecone(TEST_INDEX) 

    # Prompt the user for their Azure CLI task
    user_task = input("Please describe your Azure CLI task: ").strip()
    logging.debug("User task: %s", user_task)

    # Perform similarity search to get context using only TEST_INDEX1 for the initial response
    context = perform_similarity_search(user_task, embeddings_model, index_test1)

    # Generate the initial Azure CLI command using the similarity search context
    azure_cli_command = generate_azure_cli_command_with_context(user_task, llm, context)
    print("\nGenerated Azure CLI Command:")
    print(azure_cli_command)

    # Get user feedback and refine the command if needed
    user_feedback = get_user_feedback()
    while user_feedback != "yes":
        clarification = input("What would you like to change in the command?: ").strip()
        # Use the appropriate index for refinement
        context = perform_similarity_search(clarification, embeddings_model, index_test1)
        azure_cli_command = generate_azure_cli_command_with_context(clarification, llm, context)
        print("\nRefined Azure CLI Command:")
        print(azure_cli_command)
        
        # Ask for feedback again to see if the user is satisfied
        user_feedback = get_user_feedback()

    # Output the final command after user approval
    print("\nFinal Azure CLI Command:")
    print(azure_cli_command)
    logging.debug("User approved the Azure CLI command.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("An error occurred: %s", e)
        print(f"An error occurred: {e}")
