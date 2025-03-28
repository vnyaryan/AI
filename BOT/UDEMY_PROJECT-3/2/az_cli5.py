import os  
import logging  
import time
import openai
from openai import RateLimitError, APIError
from langchain_openai.embeddings import OpenAIEmbeddings  
from langchain_chroma import Chroma
from langchain_core.documents import Document  # Document class to create and manage documents

# Variables
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-3\2'
log_filename = 'chroma_embedding.log'
api_key = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"
chunks_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-3\2\chunk_details'  # Updated directory
chroma_collection_name = "1st_embeddings"
persist_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-3\2\chroma_persist'

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
            # Log rate limit error and retry after specified time
            retry_after = int(e.headers.get("Retry-After", delay))  # Get Retry-After time if present
            logging.error(f"Rate limit hit. Retrying after {retry_after} seconds.")
            print(f"Rate limit hit. Retrying after {retry_after} seconds.")
            time.sleep(retry_after)  # Sleep for the retry-after time
            delay = retry_after  # Update delay with the retry-after value
        except APIError as e:
            logging.error(f"APIError encountered: {e}")
            raise e  # Exit if any other API error occurs

# Initialize Chroma Vector Store
def initialize_chroma(embeddings_model):
    """Initialize the Chroma vector store with the specified collection."""
    try:
        vector_store = Chroma(
            collection_name=chroma_collection_name,
            embedding_function=embeddings_model,
            persist_directory=persist_directory
        )
        logging.debug("Chroma vector store initialized.")
        print("Chroma vector store initialized.")
        return vector_store
    except Exception as e:
        logging.error(f"Failed to initialize Chroma: {e}")
        raise

# Process each chunk file individually and store embeddings
def process_and_store_embeddings(chunks_directory, vector_store, embeddings_model):
    """Process each chunk file, generate embeddings, and store them in Chroma."""
    try:
        for filename in os.listdir(chunks_directory):
            file_path = os.path.join(chunks_directory, filename)
            chunk_name = os.path.splitext(filename)[0]  # Use the filename (without extension) as the chunk name

            # Read content of the chunk file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()  # Read and strip content to remove extra spaces/newlines

            if content:  # Process only non-empty files
                message = f"Processing chunk from file: {filename}"  # Include file name in the log
                logging.debug(message)
                print(message)

                # Generate embeddings with retry logic
                embeddings = generate_embeddings_with_retry(content, embeddings_model)

                # Store embeddings in the vector store
                vector_store.add_documents(documents=[Document(page_content=content)], ids=[chunk_name])
                logging.debug(f"Embeddings stored successfully for chunk from file {filename}.")
                print(f"Embeddings stored successfully for chunk from file {filename}.")
            else:
                logging.error(f"File {filename} is empty or invalid.")
    except Exception as e:
        logging.error(f"Failed to process and store embeddings: {e}")
        raise

# Main function
def main():
    setup_logging()
    embeddings_model = initialize_embeddings(api_key)

    # Initialize Chroma vector store
    vector_store = initialize_chroma(embeddings_model)

    # Process each chunk file and store embeddings
    process_and_store_embeddings(chunks_directory, vector_store, embeddings_model)

if __name__ == "__main__":
    main()
