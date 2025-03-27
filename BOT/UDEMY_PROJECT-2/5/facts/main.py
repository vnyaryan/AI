"""
This script handles the processing of text documents for embedding generation using OpenAI's API and stores them in a Chroma vector database.
The process includes logging, text splitting, encoding detection, and saving the results.

Logic of the script:
1. Setup logging to capture all steps and potential errors.
2. Define necessary variables such as file paths and API keys:
   - File paths are set for the input text, the output chunks, and the embeddings.
   - API key for OpenAI is defined.
3. Initialize the OpenAI embedding model using the provided API key.
4. Configure the text splitter:
   - The text splitter is configured to split the document into smaller chunks based on newline, period, and space.
5. Detect the encoding of the input text file:
   - The file's encoding is detected to ensure correct reading and conversion to UTF-8.
6. Read and convert the input text file to UTF-8:
   - The text file is read with the detected encoding and saved as a UTF-8 file for consistent processing.
7. Load and split the document into chunks:
   - The UTF-8 encoded document is loaded and split into smaller chunks using the text splitter.
8. Save the text chunks to a specified file:
   - Each chunk of text is written to an output file with a label indicating the chunk number.
9. Generate embeddings for each text chunk:
   - The OpenAI embedding model is used to generate embeddings for each text chunk.
10. Store the generated embeddings in a Chroma vector database.

Required input parameters:
1. api_key: The OpenAI API key used to access the embedding service.
2. file_path: The path to the input text file that needs to be processed.
3. chunk_file_path: The path where the split text chunks will be saved.
4. chroma_collection_name: The name of the Chroma collection where embeddings will be stored.
"""



# Provides a way of using operating system-dependent functionality, such as file and directory manipulation.
import os  

# Enables logging of messages to track events and errors that occur during the execution of the script.
import logging  

# A universal encoding detector, used here to detect the encoding of text files.
import chardet  

# Importing components from the LangChain library for processing and embedding text.

# Interface to interact with OpenAI's language models (not used directly in this script but imported for potential use).
from langchain_openai import ChatOpenAI  

# Splits text into chunks based on certain separators, helping manage large text documents.
from langchain.text_splitter import RecursiveCharacterTextSplitter  

# Used to load documents for processing, reading text files, and preparing them for further operations.
from langchain_community.document_loaders import TextLoader  

# Generates embeddings for text using OpenAI's embedding models.
from langchain_openai.embeddings import OpenAIEmbeddings  

# Import Chroma for storing and retrieving embeddings.
from langchain_chroma import Chroma
from langchain_core.documents import Document  # Document class to create and manage documents



#  variables
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\5'
log_filename = 'langchain.log'
api_key = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"
file_path = r"C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\5\facts\facts.txt"
chunk_file_path = r"C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\5\facts\chunk.txt"
chroma_collection_name = "1st_embeddings" 
persist_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\5\chroma_persist' 
query = "How to log in to Azure using az login?" 

# Function 
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

def initialize_embeddings(api_key):
    """Initialize the embedding model using the provided API key."""
    try:
        embeddings_model = OpenAIEmbeddings(api_key=api_key)
        logging.debug("Embedding model initialized.")
        return embeddings_model
    except Exception as e:
        logging.error(f"Failed to initialize the embedding model: {e}")
        raise

def configure_text_splitter():
    """Configure and return the text splitter."""
    text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "], 
    chunk_size=400,  # You can increase the chunk size slightly to avoid breaking sentences
    chunk_overlap=0  # No need for overlap if chunks are meaningful units
    )

    logging.debug("Text splitter configured.")
    return text_splitter

def detect_encoding(file_path):
    """Detect the encoding of a file using chardet."""
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        logging.debug(f"Detected encoding: {result['encoding']} for file: {file_path}")
        return result['encoding']
    except Exception as e:
        logging.error(f"Error detecting encoding for file {file_path}: {e}")
        raise

def read_and_convert_file(file_path, encoding):
    """Read a file with detected encoding and convert it to UTF-8."""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            text = f.read()
        logging.debug("File read successfully with detected encoding.")
        utf8_file_path = os.path.splitext(file_path)[0] + "_utf8.txt"
        with open(utf8_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logging.debug(f"File saved successfully in UTF-8 format: {utf8_file_path}")
        return utf8_file_path
    except Exception as e:
        logging.error(f"Failed to read or save the file: {e}")
        raise

def load_and_split_document(utf8_file_path, text_splitter):
    """Load the UTF-8 encoded file and split it into chunks."""
    try:
        loader = TextLoader(utf8_file_path)
        documents = loader.load_and_split(text_splitter=text_splitter)
        logging.debug("Documents loaded and split successfully.")
        return documents
    except Exception as e:
        logging.error(f"Failed to load and split the document: {e}")
        raise

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

def generate_and_store_embeddings(documents, vector_store):
    """Generate embeddings for each document chunk and store them in Chroma."""
    try:
        ids = [f"chunk_{i+1}" for i in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=ids)
        logging.debug("All embeddings stored successfully in Chroma.")
    except Exception as e:
        logging.error(f"Failed to generate and store embeddings: {e}")
        raise

def format_message(document, score):
    """Format the message to include separators and structured output."""
    message = f"Found document: {document.page_content} with score {score}"
    separator = "=" * 90  # Line of "=" characters for separation
    formatted_message = f"{separator}\n{message}\n{separator}"
    return formatted_message

def perform_similarity_search(query, vector_store, embeddings_model):
    """Convert the query to an embedding and perform a similarity search in Chroma."""
    try:
        query_embedding = embeddings_model.embed_documents([query])[0]
        results = vector_store.similarity_search_by_vector_with_relevance_scores(query_embedding, k=5)
        
        logging.debug("Similarity search performed successfully.")
        
        # Sort the results by relevance score (lowest score first)
        results = sorted(results, key=lambda x: x[1])  # x[1] is the score
        
        # Get the most relevant document (lowest score)
        most_relevant_document, best_score = results[0]
        
        # Format and display the most relevant document
        formatted_message = format_message(most_relevant_document, best_score)
        logging.info(formatted_message)  # Log the formatted message
        print(formatted_message)  # Print the formatted message to the screen
        
        return most_relevant_document, best_score  # Return only the top result
    except Exception as e:
        logging.error(f"Failed to perform similarity search: {e}")
        raise





def main():
    # Setup logging
    setup_logging()

    logging.debug("API key loaded successfully.")

    # Initialize embedding model
    embeddings_model = initialize_embeddings(api_key)

    # Configure text splitter
    text_splitter = configure_text_splitter()

    logging.debug(f"File path set to: {file_path}")

    # Detect file encoding
    encoding = detect_encoding(file_path)

    # Read and convert file to UTF-8
    utf8_file_path = read_and_convert_file(file_path, encoding)

    # Load and split the document
    documents = load_and_split_document(utf8_file_path, text_splitter)

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
