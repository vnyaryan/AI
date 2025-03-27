"""
Script Logic:
--------------
This script is designed to generate and store embeddings for Azure CLI command text files using OpenAI's embedding model 
and Chroma's vector store. After storing the embeddings, the script allows you to perform a similarity search based on a user query. 
The most relevant command will be returned based on the similarity score of the embedding vectors.

Steps:
1. Initialize logging to track the script execution.
2. Initialize the OpenAI embedding model using the provided API key.
3. Initialize the Chroma vector store to store and retrieve embeddings.
4. Generate embeddings for all `.txt` command files in the specified directory.
5. Store the embeddings in Chroma for future similarity searches.
6. Perform a similarity search based on a query (example provided) and return the most relevant command and its similarity score.

Input Parameters:
-----------------
- log_directory: The directory where logs will be saved.
- log_filename: The filename for the log file.
- commands_directory: The directory where Azure CLI command text files are located.
- chroma_collection_name: The name of the Chroma collection for storing embeddings.
- persist_directory: The directory to persist the Chroma vector store data.
- api_key: OpenAI API key for generating embeddings.

Output Details:
---------------
- A log file will be created in the specified log_directory, which will contain detailed logs of the script execution.
- Embeddings will be generated for each text file in the commands_directory and stored in Chroma.
- The similarity search will output the most relevant command text based on the input query and its similarity score.
"""

import os
import logging
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Constants for logging, embedding, and Chroma setup
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\7\facts\logs'
log_filename = 'cli_processing.log'
commands_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\7\facts\commands'  
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

# Initialize Chroma store
def initialize_chroma(embeddings_model):
    try:
        vector_store = Chroma(
            collection_name=chroma_collection_name,
            embedding_function=embeddings_model,
            persist_directory=persist_directory
        )
        logging.debug("Chroma vector store initialized.")
        return vector_store
    except Exception as e:
        logging.error(f"Failed to initialize Chroma: {e}")
        raise

# Generate and store embeddings in Chroma
def generate_and_store_embeddings(embeddings_model, vector_store, commands_directory):
    try:
        files = [f for f in os.listdir(commands_directory) if f.endswith('.txt')]
        if not files:
            logging.error("No command files found for embedding generation.")
            raise ValueError("No command files found for embedding generation.")

        for file in files:
            file_path = os.path.join(commands_directory, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            doc_id = file.split('.')[0]  # Use filename without extension as document ID
            document = Document(page_content=content)  # Wrap content in Document object
            embedding = embeddings_model.embed_documents([content])[0]
            vector_store.add_documents([document], ids=[doc_id])  # Pass Document object
            logging.debug(f"Embedding generated and stored for file: {file}")

    except Exception as e:
        logging.error(f"Failed to generate and store embeddings: {e}")
        raise

# Perform similarity search
def perform_similarity_search(query, vector_store, embeddings_model):
    try:
        query_embedding = embeddings_model.embed_documents([query])[0]
        results = vector_store.similarity_search_by_vector_with_relevance_scores(query_embedding, k=5)

        results = sorted(results, key=lambda x: x[1])  # Sort by relevance score
        most_relevant_document, best_score = results[0]

        logging.info(f"Most relevant document: {most_relevant_document.page_content} with score {best_score}")
        print(f"Most relevant document: {most_relevant_document.page_content} with score {best_score}")
        return most_relevant_document, best_score
    except Exception as e:
        logging.error(f"Failed to perform similarity search: {e}")
        raise

# Main function to execute the embedding and search flow
def main():
    setup_logging()
    logging.info("Starting embedding generation and storage...")

    # Initialize embeddings model
    embeddings_model = initialize_embeddings(api_key)

    # Initialize Chroma vector store
    vector_store = initialize_chroma(embeddings_model)

    # Generate and store embeddings for command files
    generate_and_store_embeddings(embeddings_model, vector_store, commands_directory)

    # Example query for similarity search
    query = "How to log in to Azure using az login?"
    perform_similarity_search(query, vector_store, embeddings_model)

    logging.info("Processing completed.")

if __name__ == "__main__":
    main()
