import os
import logging
from openai import OpenAI

# Define the API key as a constant
api_key = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Define logging directory and filename
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\14\log'
log_filename = 'phase2.log'

# Define vector store name, assistant name, and file paths as constants
vector_store_name = "VectorStore3"
assistant_name = "My Assistant3"
file_paths = [r"C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\14\azure_module.pdf"]

# Initialize OpenAI client with the API key
client = OpenAI(api_key=api_key)

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

# Function to create an assistant and attach a vector store
def create_assistant_and_vector_store():
    try:
        logging.debug("Creating assistant with file_search tool enabled...")
        assistant = client.beta.assistants.create(
            name=assistant_name,
            instructions="Use the vector store to answer queries based on uploaded files.",
            model="gpt-3.5-turbo",
            tools=[{"type": "file_search"}]
        )
        logging.debug(f"Assistant created with ID: {assistant.id}")

        # Step 2: Create a new vector store
        logging.debug("Creating vector store...")
        vector_store = client.beta.vector_stores.create(name=vector_store_name)
        logging.debug(f"Vector store created with ID: {vector_store.id}")

        # Step 3: Prepare the file streams for upload
        file_streams = [open(path, "rb") for path in file_paths]

        # Step 4: Upload files to the vector store and poll the status
        logging.debug("Uploading files to vector store...")
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=file_streams
        )
        
        logging.debug(f"Upload Status: {file_batch.status}")
        logging.debug(f"Files Processed: {file_batch.file_counts}")

        # Step 5: Attach the vector store to the assistant
        logging.debug("Attaching vector store to assistant...")
        assistant = client.beta.assistants.update(
            assistant_id=assistant.id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
        )
        
        logging.debug(f"Assistant {assistant.id} successfully updated with Vector Store {vector_store.id}")
        
        return assistant.id, vector_store.id

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return None, None

# Function to perform a similarity search and display relevant content
def perform_similarity_search(assistant_id, vector_store_id, user_query):
    try:
        logging.debug("Performing similarity search...")
        
        # Create a thread for the query
        thread = client.beta.threads.create(
            messages=[
                {"role": "user", "content": user_query}
            ],
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
        )
        logging.debug(f"Thread created with ID: {thread.id}")
        
        # Run the query on the thread and capture only the relevant content
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant_id,
            instructions="Search for similar content in the vector store"
        ) as stream:
            for event in stream:
                # Check if the event contains a message and extract its content
                if hasattr(event.data, 'message') and hasattr(event.data.message, 'content'):
                    for content_block in event.data.message.content:
                        if content_block.type == 'text':
                            response_text = content_block.text.value
                            print(response_text)
                            logging.debug(f"Query Response: {response_text}")
            
    except Exception as e:
        logging.error(f"An error occurred during similarity search: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run the process to create an assistant and upload files to vector store
    assistant_id, vector_store_id = create_assistant_and_vector_store()

    if assistant_id and vector_store_id:
        logging.debug(f"Process completed successfully. Assistant ID: {assistant_id}, Vector Store ID: {vector_store_id}")
        
        # Accept a user query
        user_query = input("Enter your query: ")

        # Perform the similarity search and display relevant content
        perform_similarity_search(assistant_id, vector_store_id, user_query)
    else:
        logging.error("Process failed.")
