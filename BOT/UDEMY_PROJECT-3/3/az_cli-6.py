import os
import logging
import sys
from pinecone import Pinecone, ServerlessSpec
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

# Constants for logging and API setup
LOG_DIRECTORY = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\UDEMY_PROJECT-3\3'
LOG_FILENAME = 'az_cli-6.log'
API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"
PINECONE_API_KEY = "9fe3c832-5827-45f6-b066-e740b9b13e33"
PINECONE_ENVIRONMENT = "us-east-1"
TEST_INDEX1 = "testindex1"
#TEST_INDEX = "testindex"

# Truncate the context to a specified number of characters before using it in the prompt
def truncate_context(context, max_chars=2000):
    """Truncate the context to the specified maximum number of characters."""
    return context[:max_chars]

# Set up logging configuration to track script execution
def setup_logging():
    try:
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(LOG_DIRECTORY, LOG_FILENAME),
            filemode='w'
        )
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        sys.exit(1)

# Initialize Pinecone
def initialize_pinecone(api_key, environment, index_name):
    """Initialize the Pinecone instance and retrieve the index."""
    try:
        # Create Pinecone instance
        pc = Pinecone(api_key=api_key)

        # Check if the index exists
        if index_name not in pc.list_indexes().names():
            print(f"Index '{index_name}' not found. Creating a new index...")
            pc.create_index(
                name=index_name,
                dimension=1536,  # Update the dimension based on your embeddings
                metric='cosine',
                spec=ServerlessSpec(cloud="aws", region=environment)
            )
        index = pc.Index(index_name)
        logging.debug(f"Pinecone index '{index_name}' initialized.")
        return index
    except Exception as e:
        logging.error(f"Failed to initialize Pinecone: {e}")
        raise

# Perform similarity search to get additional context for the user's query
def perform_similarity_search(user_task, embeddings_model, index):
    # Generate the embedding for the user query
    embedding = embeddings_model.embed_documents([user_task])[0]
    
    # Log the embedding details
    logging.debug(f"Generated embedding for query '{user_task}': {embedding}")
    try:
        results = index.query(vector=embedding, top_k=3, include_values=True)
        results = sorted(results['matches'], key=lambda x: x['score'])
        additional_context = "\n".join([str(result['values']) for result in results[:3]])
        logging.debug("Similarity search results obtained successfully.")
        return additional_context
    except Exception as e:
        logging.error(f"Failed to perform similarity search: {e}")
        raise

# Generate an Azure CLI command using the context from similarity search
def generate_azure_cli_command_with_context(user_task, llm, context, memory):
    logging.debug("Generating Azure CLI command using context from similarity search.")
    system_template = "You are an expert of Azure and az cli. Use the following context to generate an appropriate Azure az cli command as per the user’s task."
    human_template = "User task: {user_task}\nContext: {context}"

    # Truncate context to fit within token limits
    truncated_context = truncate_context(context, max_chars=2000)

    system_prompt_template = SystemMessagePromptTemplate.from_template(system_template)
    human_prompt_template = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt_template = ChatPromptTemplate.from_messages([system_prompt_template, human_prompt_template])

    formatted_prompt = chat_prompt_template.format_prompt(user_task=user_task, context=truncated_context)
    messages = formatted_prompt.to_messages()

    serialized_messages = [{"role": "system", "content": msg.content} if isinstance(msg, SystemMessage)
                           else {"role": "user", "content": msg.content} for msg in messages]

    response = llm.invoke(serialized_messages)

    memory.save_context({'input': serialized_messages}, {'output': response.content})
    return response.content

# Main function to drive the interaction with the user
def main():
    setup_logging()

    llm = ChatOpenAI(openai_api_key=API_KEY)
    memory = ConversationSummaryMemory(memory_key="messages", return_messages=True, llm=llm)

    embeddings_model = OpenAIEmbeddings(api_key=API_KEY)

    # Initialize Pinecone indices
    index_test1 = initialize_pinecone(PINECONE_API_KEY, PINECONE_ENVIRONMENT, TEST_INDEX1)
    #index_test = initialize_pinecone(PINECONE_API_KEY, PINECONE_ENVIRONMENT, TEST_INDEX)

    user_task = input("Please describe your Azure CLI task: ").strip()
    logging.debug("User task: %s", user_task)

    context = perform_similarity_search(user_task, embeddings_model, index_test1)

    azure_cli_command = generate_azure_cli_command_with_context(user_task, llm, context, memory)
    print("\nGenerated Azure CLI Command:")
    print(azure_cli_command)

    user_feedback = input("\nIs the generated Azure CLI command correct? (yes/no): ").strip().lower()
    while user_feedback != "yes":
        clarification = input("What would you like to change or clarify in the command?: ").strip()
        azure_cli_command = refine_azure_cli_command_new(clarification, llm, memory, index_test, embeddings_model)
        print("\nRefined Azure CLI Command:")
        print(azure_cli_command)
        user_feedback = input("\nIs the refined command correct? (yes/no): ").strip().lower()

    print("\nFinal Azure CLI Command:")
    print(azure_cli_command)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("An error occurred: %s", e)
        print(f"An error occurred: {e}")
