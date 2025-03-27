"""
This script handles the processing of text documents for embedding generation using OpenAI's API.
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
10. Save the generated embeddings to a specified file:
    - The embeddings for each chunk are saved to an output file with corresponding labels.

Required input parameters:
1. api_key: The OpenAI API key used to access the embedding service.
2. file_path: The path to the input text file that needs to be processed.
3. chunk_file_path: The path where the split text chunks will be saved.
4. output_file_path: The path where the generated embeddings will be saved.
"""
# Provides a way of using operating system-dependent functionality, such as file and directory manipulation.
import os  

# Enables logging of messages to track events and errors that occur during the execution of the script.
import logging 

# A universal encoding detector, used here to detect the encoding of text files.
import chardet  

# Importing components from the LangChain library for processing and embedding text.

# Splits text into chunks based on certain separators, helping manage large text documents.
from langchain.text_splitter import RecursiveCharacterTextSplitter  

# Used to load documents for processing, reading text files, and preparing them for further operations.
from langchain_community.document_loaders import TextLoader  

# Generates embeddings for text using OpenAI's embedding models.
from langchain_openai.embeddings import OpenAIEmbeddings

# Define all variables before the functions
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\2'
log_filename = 'langchain.log'
api_key = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"
file_path = r"C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\2\facts\facts.txt"
chunk_file_path = r"C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\2\facts\chunk.txt"
output_file_path = r"C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\2\facts\embedding.txt"

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
        separators=["\n", ".", " "], 
        chunk_size=200,
        chunk_overlap=20  
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

def generate_and_save_embeddings(documents, embeddings_model, output_file_path):
    """Generate embeddings for each document chunk and save them."""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for i, document in enumerate(documents):
                embedding = embeddings_model.embed_documents([document.page_content])
                f.write(f"Chunk {i + 1}:\n")
                f.write(f"Embedding: {embedding}\n\n")
                logging.debug(f"Embedding {i + 1} generated and saved.")
        logging.debug(f"All embeddings saved successfully to {output_file_path}.")
    except Exception as e:
        logging.error(f"Failed to generate and save embeddings: {e}")
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

    # Generate and save embeddings
    generate_and_save_embeddings(documents, embeddings_model, output_file_path)

if __name__ == "__main__":
    main()
