"""
This script processes text chunks, generates embeddings using the OpenAI model, and stores them in a Pinecone index for efficient retrieval.

Logic of the script:
1. Sets up logging configuration to record script activities and potential errors.
2. Initializes the OpenAI embeddings model using a provided API key.
3. Initializes a Pinecone index, creating it if it does not already exist.
4. Reads the list of chunk file names from a specified file.
5. For each chunk file, reads its content and generates embeddings with retry logic for handling rate limits.
6. Stores the generated embeddings in the Pinecone index with the chunk file name as the document ID.
7. Logs the success or failure of each operation for troubleshooting and monitoring.

Input Parameters:
1. api_key: A constant representing the OpenAI API key for accessing the embedding model.
2. pinecone_api_key: A constant representing the Pinecone API key for accessing the Pinecone service.
3. pinecone_environment: A constant specifying the environment for Pinecone.
4. index_name: A constant representing the name of the Pinecone index to use.
5. chunks_directory: A constant specifying the directory containing the text chunks.
6. names_file: A constant specifying the path to the file containing the names of the chunk files to be processed.
7. log_directory: A constant representing the directory where log files will be stored.
8. log_filename: A constant representing the log file's name.

Output Details:
1. Pinecone Index: Stores embeddings for each text chunk with document IDs corresponding to the chunk file names.
2. az_cli8.log: Records detailed logs of the script execution, including successes, failures, and any rate limit handling during embedding generation.
3. Console Output: Displays progress messages and errors during execution to provide real-time feedback.
"""

import os
import logging
import time
import openai
import pinecone
from pinecone import Pinecone, ServerlessSpec
from openai import RateLimitError, APIError
from langchain_openai.embeddings import OpenAIEmbeddings 
from langchain.docstore.document import Document

# Variables
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-3\2\log'
log_filename = "az_cli8.log"
api_key = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"
pinecone_api_key = "9fe3c832-5827-45f6-b066-e740b9b13e33"
pinecone_environment = "us-east-1"
index_name = "testindex1"
chunks_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-3\2\az_cli_summary'
names_file = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-3\2\name.txt'

# Function for logging setup
def setup_logging():
    """Setup logging configuration with a fixed log filename."""
    try:
        os.makedirs(log_directory, exist_ok=True)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='w')
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")

# Initialize OpenAI Embeddings
def initialize_embeddings(api_key):
    """Initialize the embedding model using the provided API key."""
    try:
        embeddings_model = OpenAIEmbeddings(api_key=api_key)
        logging.debug("Embedding model initialized.")
        print("Embedding model initialized.")
        return embeddings_model
    except Exception as e:
        logging.error(f"Failed to initialize the embedding model: {e}")
        raise

# Generate embeddings with retry logic for rate limits
def generate_embeddings_with_retry(text, embeddings_model, retries=5):
    """Generate embeddings with retry in case of rate limit errors."""
    delay = 10  # Initial delay of 10 seconds
    for attempt in range(retries):
        try:
            embeddings = embeddings_model.embed_documents([text])  # Generate embedding for the chunk
            logging.debug(f"Embedding generated for text chunk.")
            return embeddings
        except RateLimitError as e:
            retry_after = int(e.headers.get("Retry-After", delay))  # Get Retry-After time if present
            logging.error(f"Rate limit hit. Retrying after {retry_after} seconds.")
            print(f"Rate limit hit. Retrying after {retry_after} seconds.")
            time.sleep(retry_after)
            delay = retry_after  # Update delay with the retry-after value
        except APIError as e:
            logging.error(f"APIError encountered: {e}")
            raise e  # Exit if any other API error occurs



# Initialize Pinecone Index
def initialize_pinecone():
    """Initialize the Pinecone index."""
    try:
        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=pinecone_api_key)
        
        # Create the index if it doesn't exist
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,  # Dimension should match the embedding size
                metric='euclidean',  # Or 'euclidean' depending on your use case
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )



        # Retrieve the index
        index = pc.Index(index_name)
        logging.debug("Pinecone index initialized.")
        print("Pinecone index initialized.")
        return index
    except Exception as e:
        logging.error(f"Failed to initialize Pinecone: {e}")
        raise


def process_and_store_embeddings(chunks_directory, pinecone_index, embeddings_model, names_file):
    """Process each chunk file from names.txt individually, generate embeddings, and store them in Pinecone."""
    try:
        # Open names.txt file to get the chunk names
        with open(names_file, 'r', encoding='utf-8') as file:
            chunk_names = [line.strip() for line in file.readlines() if line.strip()]  # Read all chunk names

        # Process each chunk individually
        for chunk_name in chunk_names:
            # Get the full path to the chunk file
            file_path = os.path.join(chunks_directory, chunk_name)

            # Check if the chunk file exists before processing
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as chunk_file:
                    content = chunk_file.read().strip()  # Read and strip content to remove extra spaces/newlines

                if content:  # Process only non-empty files
                    message = f"Processing chunk from file: {chunk_name}"  # Include file name in the log
                    logging.debug(message)
                    print(message)

                    # Generate embeddings with retry logic
                    embeddings = generate_embeddings_with_retry(content, embeddings_model)

                    # Use the chunk name (including .txt extension) as the document ID
                    doc_id = chunk_name

                    # Store embeddings in Pinecone
                    pinecone_index.upsert([(doc_id, embeddings[0])])
                    logging.debug(f"Embeddings stored successfully for chunk: {chunk_name}.")
                    print(f"Embeddings stored successfully for chunk: {chunk_name}.")
                else:
                    logging.error(f"File {chunk_name} is empty or invalid.")
                    print(f"File {chunk_name} is empty or invalid.")
            else:
                logging.error(f"Chunk file {chunk_name} does not exist in {chunks_directory}.")
                print(f"Chunk file {chunk_name} does not exist in {chunks_directory}.")
    except Exception as e:
        logging.error(f"Failed to process and store embeddings: {e}")
        raise


# Main function
def main():
    setup_logging()
    embeddings_model = initialize_embeddings(api_key)

    # Initialize Pinecone index
    pinecone_index = initialize_pinecone()

    # Process each chunk file and store embeddings in Pinecone
    process_and_store_embeddings(chunks_directory, pinecone_index, embeddings_model, names_file)

if __name__ == "__main__":
    main()
