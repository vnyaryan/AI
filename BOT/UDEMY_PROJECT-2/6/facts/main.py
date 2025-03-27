"""
Script Logic:
This script processes a PDF document, splits the content into smaller chunks, generates embeddings for each chunk, 
and stores them in a Chroma vector store. It then allows performing a similarity search against the stored document embeddings 
using a query string. The results of the similarity search are displayed on the screen.

Input Parameters:
1. api_key: OpenAI API key to initialize the embeddings model.
2. file_path: Path to the PDF file that will be processed and split into chunks.
3. chunk_file_path: Path to save the text chunks extracted from the PDF file.
4. chroma_collection_name: The name of the collection used in the Chroma vector store.
5. persist_directory: Directory to persist the Chroma vector store data.
6. query: The query string used to perform similarity search on the document embeddings.

Output:
1. A log file ('langchain.log') that contains debug and error logs of the script execution.
2. A text file ('az_cli.txt') that stores the document chunks extracted from the PDF.
3. A Chroma vector store that holds the embeddings of the document chunks.
4. Console output showing the top 5 documents that are most similar to the input query along with their relevance scores.
"""

import os  
import logging  
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_openai.embeddings import OpenAIEmbeddings  
from langchain_chroma import Chroma
from langchain_core.documents import Document  # Document class to create and manage documents

#  Variables
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\6'
log_filename = 'langchain.log'
api_key = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"
file_path = r"C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\6\facts\az_cli.pdf"  
chunk_file_path = r"C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\6\facts\az_cli.txt"
chroma_collection_name = "1st_embeddings" 
persist_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\6\chroma_persist' 
query = "tell me about az login" 

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
        return embeddings_model
    except Exception as e:
        logging.error(f"Failed to initialize the embedding model: {e}")
        raise

# Text splitter configuration
def configure_text_splitter():
    """Configure and return the text splitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", ".", " "], 
        chunk_size=200,
        chunk_overlap=20  
    )
    logging.debug("Text splitter configured.")
    return text_splitter

# Load and process PDF using PyPDFLoader
def load_and_split_pdf(file_path, text_splitter):
    """Load the PDF file using PyPDFLoader and split it into chunks."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split(text_splitter=text_splitter)
        logging.debug("PDF loaded and split successfully.")
        return documents
    except Exception as e:
        logging.error(f"Failed to load and split the PDF: {e}")
        raise

# Save text chunks
def save_chunks(documents, chunk_file_path):
    """Save the document chunks to a text file."""
    try:
        with open(chunk_file_path, 'w', encoding='utf-8') as chunk_file:
            for i, document in enumerate(documents):
                chunk_file.write(f"Chunk {i + 1}:\n{document.page_content}\n\n")
                logging.debug(f"Chunk {i + 1} saved to {chunk_file_path}.")
    except Exception as e:
        logging.error(f"Failed to save chunks to file: {e}")
        raise

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
        return vector_store
    except Exception as e:
        logging.error(f"Failed to initialize Chroma: {e}")
        raise

# Generate and store embeddings
def generate_and_store_embeddings(documents, vector_store):
    """Generate embeddings for each document chunk and store them in Chroma."""
    try:
        ids = [f"chunk_{i+1}" for i in range(len(documents))]
        
        # Check if the documents have valid content
        for i, document in enumerate(documents):
            if not document.page_content.strip():
                logging.error(f"Document chunk {i+1} is empty.")
            else:
                logging.debug(f"Document chunk {i+1} content: {document.page_content[:100]}...")  # Log first 100 chars
        
        if len(ids) == 0:
            logging.error("No valid document chunks were found.")
            raise ValueError("No valid document chunks to generate embeddings.")
        
        vector_store.add_documents(documents=documents, ids=ids)
        logging.debug("All embeddings stored successfully in Chroma.")
    except Exception as e:
        logging.error(f"Failed to generate and store embeddings: {e}")
        raise


# Perform similarity search
def perform_similarity_search(query, vector_store, embeddings_model):
    """Convert the query to an embedding and perform a similarity search in Chroma."""
    try:
        query_embedding = embeddings_model.embed_documents([query])[0]
        results = vector_store.similarity_search_by_vector_with_relevance_scores(query_embedding, k=5)
        
        logging.debug("Similarity search performed successfully.")
        for document, score in results:
            formatted_message = f"Found document: {document.page_content} with score {score}"
            print(formatted_message)  # Print the formatted message to the screen
        return results
    except Exception as e:
        logging.error(f"Failed to perform similarity search: {e}")
        raise

# Main function
def main():
    setup_logging()
    embeddings_model = initialize_embeddings(api_key)
    text_splitter = configure_text_splitter()

    # Load and split the PDF
    documents = load_and_split_pdf(file_path, text_splitter)

    # Save the chunks to a text file
    save_chunks(documents, chunk_file_path)

    # Initialize Chroma vector store
    vector_store = initialize_chroma(embeddings_model)

    # Generate and store embeddings in Chroma
    generate_and_store_embeddings(documents, vector_store)

    # Perform a similarity search using the query
    perform_similarity_search(query, vector_store, embeddings_model)

if __name__ == "__main__":
    main()
