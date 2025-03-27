import logging
from openai import OpenAI
import os

# Define the API key as a constant
api_key = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Define logging directory and filename
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\14\log'
log_filename = 'phase2.log'

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

# Function to create an assistant
def create_assistant():
    logging.debug("Creating assistant...")
    assistant = client.beta.assistants.create(
        name="Query Assistant",
        instructions="Use the vector store to answer user queries.",
        model="gpt-3.5-turbo",  # Compatible model
        tools=[{"type": "file_search"}]
    )
    logging.debug(f"Assistant created with ID: {assistant.id}")
    return assistant.id

# Function to perform similarity search using a thread and a run
def perform_similarity_search(vector_store_id, user_query, assistant_id):
    try:
        logging.debug("Performing similarity search...")
        
        # Create a thread for the query
        thread = client.beta.threads.create(
            messages=[
                {"role": "user", "content": user_query}
            ],
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
        )
        
        # Run the query on the thread
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant_id,
            instructions="Search for similar content in the vector store"
        ) as stream:
            for event in stream:
                print(event)
                logging.debug(event)
            
    except Exception as e:
        logging.error(f"An error occurred during similarity search: {str(e)}")
        print(f"An error occurred: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Define the vector store ID and query
    vector_store_id = "vs_fDDGGpMMaaqSAiXVx2PGkEwd"  # Using the actual vector store ID
    user_query = input("Enter your query: ")

    # Create the assistant
    assistant_id = create_assistant()

    # Perform the similarity search
    perform_similarity_search(vector_store_id, user_query, assistant_id)

