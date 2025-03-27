"""
Script Logic:
--------------
This script performs a similarity search on Azure CLI command embeddings stored in a Chroma vector store. 
The process involves:
1. Setting up logging to track the execution.
2. Initializing OpenAI embeddings using the provided API key.
3. Loading the existing Chroma vector store to retrieve the stored embeddings.
4. Performing a similarity search for a given query to find the most relevant document based on the embeddings.

Input Parameters:
-----------------
- log_directory: Path to the directory where logs will be saved.
- log_filename: Name of the log file.
- chroma_collection_name: The collection name for storing and retrieving embeddings in Chroma.
- persist_directory: Path to the directory where the Chroma vector store data is persisted.
- api_key: OpenAI API key for accessing the embedding model.

Output Details:
---------------
- Log file: A detailed log of the script execution will be saved in the specified log_directory.
- The similarity search will output the most relevant document content based on the input query and display its relevance score.
"""
import os
import logging
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

# Constants for logging, embedding, and Chroma setup
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\7\facts\logs'
log_filename = 'cli_processing.log'
chroma_collection_name = "azure_cli_embeddings"
persist_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\7\facts\chroma_persist'
api_key = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Function to set up logging
def setup_logging():
    os.makedirs(log_directory, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=os.path.join(log_directory, log_filename),
                        filemode='w')
    logging.debug("Logging setup completed.")

# Initialize OpenAI embeddings
def initialize_embeddings(api_key):
    try:
        embeddings_model = OpenAIEmbeddings(api_key=api_key)
        logging.debug("OpenAI embedding model initialized.")
        return embeddings_model
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI embeddings: {e}")
        raise

# Load the existing Chroma vector store
def load_chroma_store(embeddings_model):
    try:
        vector_store = Chroma(
            collection_name=chroma_collection_name,
            embedding_function=embeddings_model,
            persist_directory=persist_directory
        )
        logging.debug("Chroma vector store loaded from persist directory.")
        return vector_store
    except Exception as e:
        logging.error(f"Failed to load Chroma store: {e}")
        raise

# Perform similarity search for a query
def perform_similarity_search(query, vector_store, embeddings_model):
    try:
        # Create embedding for the query
        query_embedding = embeddings_model.embed_documents([query])[0]

        # Perform similarity search
        results = vector_store.similarity_search_by_vector_with_relevance_scores(query_embedding, k=5)

        # Sort and get the most relevant result
        results = sorted(results, key=lambda x: x[1])  # Sort by relevance score
        most_relevant_document, best_score = results[0]

        logging.info(f"Most relevant document: {most_relevant_document.page_content} with score {best_score}")
        print(f"Most relevant document: {most_relevant_document.page_content} with score {best_score}")
        return most_relevant_document, best_score
    except Exception as e:
        logging.error(f"Failed to perform similarity search: {e}")
        raise

# Main function to execute the search flow
def main():
    setup_logging()
    logging.info("Starting similarity search...")

    # Initialize embeddings model
    embeddings_model = initialize_embeddings(api_key)

    # Load Chroma vector store
    vector_store = load_chroma_store(embeddings_model)

    # Example query for similarity search
    query = "tell me about az login parameter"
    perform_similarity_search(query, vector_store, embeddings_model)

    logging.info("Processing completed.")

if __name__ == "__main__":
    main()
