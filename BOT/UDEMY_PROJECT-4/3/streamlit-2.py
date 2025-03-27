import os
import logging
import sys
import pinecone
import streamlit as st  # Importing Streamlit for web app interface
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings

# Constants for logging and API setup
LOG_DIRECTORY = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-4\3'
LOG_FILENAME = 'streamlit-2.log'
API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"
PINECONE_API_KEY = "9fe3c832-5827-45f6-b066-e740b9b13e33"
PINECONE_ENVIRONMENT = "us-east-1"
TEST_INDEX1 = "testindex1"
TEST_INDEX = "testindex"

# Set up logging configuration to track script execution
def setup_logging():
    try:
        # Ensure the log directory exists
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(LOG_DIRECTORY, LOG_FILENAME),
            filemode='a'  # Append mode
        )
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        sys.exit(1)

# Truncate the context to a specified number of characters before using it in the prompt
def truncate_context(context, max_chars=2000):
    """Truncate the context to the specified maximum number of characters."""
    return context[:max_chars]

# Perform similarity search to get additional context for the user's query
def perform_similarity_search(user_task, embeddings_model, index):
    logging.debug("Generating embedding for similarity search.")
    embedding = embeddings_model.embed_documents([user_task])[0]
    try:
        # Perform similarity search on the specified index
        results = index.query(vector=embedding, top_k=3, include_values=True)

        # Sort results based on similarity scores
        results = sorted(results['matches'], key=lambda x: x['score'])

        # Extract the actual content from the top results and ensure it's a string
        additional_context = "\n".join([str(result['values']) for result in results[:3]])
        logging.debug("Similarity search results obtained successfully.")
        return additional_context
    except Exception as e:
        logging.error(f"Failed to perform similarity search: {e}")
        raise

# Generate an Azure CLI command using the context from similarity search
def generate_azure_cli_command_with_context(user_task, llm, context, memory):
    logging.debug("Generating Azure CLI command using context from similarity search.")
    truncated_context = truncate_context(context, max_chars=2000)  # Truncate context to 2000 characters
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are an expert of Azure and az cli. Use the following context to generate an appropriate Azure az cli command as per the user’s task."
            ),
            HumanMessagePromptTemplate.from_template(
                "User task: {user_task}\nContext: {context}"
            )
        ]
    )
    azure_cli_request_prompt = chat_prompt_template.format_prompt(
        user_task=user_task,
        context=truncated_context
    ).to_messages()
    
    response = llm.invoke(azure_cli_request_prompt)
    
    # Save the interaction context using memory
    memory.save_context({'input': azure_cli_request_prompt}, {'output': response.content})
    
    return response.content

# Refine the Azure CLI command based on user feedback and let OpenAI decide if parameters are involved
def refine_azure_cli_command_new(clarification, llm, memory, index_test, embeddings_model):
    logging.debug("Refining Azure CLI command with clarification: %s", clarification)

    # Step 1: Ask OpenAI if the clarification involves parameter adjustments
    clarification_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are refining an Azure CLI command based on the user's clarification."
            ),
            HumanMessagePromptTemplate.from_template(
                "The user suggested the following clarification: {clarification}\n"
                "Does this clarification involve changes to the parameters of the Azure CLI command? Please answer 'yes' or 'no'."
            )
        ]
    )
    clarification_prompt = clarification_prompt_template.format_prompt(
        clarification=clarification
    ).to_messages()

    # Ask OpenAI to decide whether the clarification involves parameters
    openai_response = llm.invoke(clarification_prompt)
    is_parameter_related = openai_response.content.strip().lower()

    # Step 2: If OpenAI decides that parameters are involved, perform a parameter search using `perform_similarity_search`
    if is_parameter_related == "yes":
        logging.debug("OpenAI determined that the clarification involves parameters. Performing parameter search.")
        
        # Use the existing similarity search function for parameter-related clarification
        try:
            parameter_context = perform_similarity_search(clarification, embeddings_model, index_test)
            # Truncate the parameter context to a reasonable length (same as before)
            parameter_context = truncate_context(parameter_context, max_chars=2000)
            logging.debug("Parameter context from similarity search: %s", parameter_context)
        except Exception as e:
            logging.error(f"Failed to perform parameter search on index_test: {e}")
            parameter_context = ""  # If the search fails, proceed without parameter context
    else:
        # If OpenAI determines that no parameters are involved
        logging.debug("OpenAI determined that the clarification does not involve parameters.")
        parameter_context = ""

    # Step 3: Prepare the final prompt for refinement with both clarification and optional parameter context
    final_refinement_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are refining an Azure CLI command based on the user's clarification. The following additional context may be included."
            ),
            HumanMessagePromptTemplate.from_template(
                "The user suggested the following clarification: {clarification}\n"
                "Relevant parameters (if applicable): {parameter_context}"
            )
        ]
    )
    final_refinement_prompt = final_refinement_prompt_template.format_prompt(
        clarification=clarification, 
        parameter_context=parameter_context
    ).to_messages()

    # Step 4: Include conversation history from memory
    history_messages = memory.load_memory_variables({'input': 'messages'})['messages']
    refinement_request_prompt = history_messages + final_refinement_prompt
    
    # Generate the refined command based on user clarification and parameter context (if applicable)
    response = llm.invoke(refinement_request_prompt)
    
    # Step 5: Save the new interaction to memory
    memory.save_context({'input': refinement_request_prompt}, {'output': response.content})
    
    return response.content

# Initialize Pinecone Index
def initialize_pinecone(index_name):
    """Initialize the Pinecone index."""
    try:
        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        # Retrieve the index
        index = pc.Index(index_name)
        logging.debug(f"Pinecone index '{index_name}' initialized.")
        return index
    except Exception as e:
        logging.error(f"Failed to initialize Pinecone: {e}")
        raise

# Streamlit App Integration
def main():
    # Initialize logging
    setup_logging()

    st.title("Azure CLI Command Generator")

    # Initialize the LLM and memory
    llm = ChatOpenAI(openai_api_key=API_KEY)
    memory = ConversationSummaryMemory(memory_key="messages", return_messages=True, llm=llm)
    
    # Initialize Pinecone and embeddings
    embeddings_model = OpenAIEmbeddings(api_key=API_KEY)
    index_test1 = initialize_pinecone(TEST_INDEX1)

    # Initialize session state
    if 'azure_cli_command' not in st.session_state:
        st.session_state['azure_cli_command'] = None
    if 'user_feedback' not in st.session_state:
        st.session_state['user_feedback'] = ""

    # Input area for the user's Azure CLI task
    user_task = st.text_area("Describe your Azure CLI task:", value=st.session_state.get('user_task', ''))
    
    # Button to generate the initial Azure CLI command
    if st.button("Generate Azure CLI Command"):
        if user_task:
            # Store the task in session state
            st.session_state['user_task'] = user_task

            # Perform similarity search to get context using TEST_INDEX1
            context = perform_similarity_search(user_task, embeddings_model, index_test1)

            # Generate the initial Azure CLI command using the similarity search context
            azure_cli_command = generate_azure_cli_command_with_context(user_task, llm, context, memory)

            # Store the generated command in session state
            st.session_state['azure_cli_command'] = azure_cli_command

            # Display the generated Azure CLI command
            st.subheader("Generated Azure CLI Command")
            st.code(st.session_state['azure_cli_command'])
            
    # Display and allow refinement if a command is generated
    if st.session_state['azure_cli_command']:
        user_feedback = st.text_input("Would you like to refine the command? (yes/no)", value=st.session_state['user_feedback'])
        
        if user_feedback.lower() == "yes":
            clarification = st.text_area("What would you like to change or clarify in the command?", value="")
            if st.button("Submit Clarification"):
                # Refine the command based on user clarification
                refined_command = refine_azure_cli_command_new(clarification, llm, memory, index_test1, embeddings_model)
                st.session_state['azure_cli_command'] = refined_command
                st.subheader("Refined Azure CLI Command")
                st.code(st.session_state['azure_cli_command'])
        
        # Store user feedback in session state
        st.session_state['user_feedback'] = user_feedback
    else:
        st.error("Please describe your Azure CLI task.")

if __name__ == "__main__":
    main()
