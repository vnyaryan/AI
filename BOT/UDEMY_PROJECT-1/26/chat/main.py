import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationSummaryMemory

def setup_logging():
    """Setup logging configuration with a fixed log filename."""
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT\26'
        log_filename = 'langchain.log'

        # Ensure the log directory exists
        os.makedirs(log_directory, exist_ok=True)

        # Configure logging
        logging.basicConfig(level=logging.DEBUG,  # Set to DEBUG level
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='a'  # Append mode
        )
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")

# Step 1: Setup logging
setup_logging()

# Step 2: Securely manage the API key and initialize the ChatOpenAI class with it.
api_key = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"  # Placeholder API key.

# Step 3: Initialize conversation summary memory
memory = ConversationSummaryMemory(
    memory_key="messages",   # The key under which conversation messages are stored and retrieved.
    return_messages=True,    # Ensure that messages are returned as part of the response.
    llm=ChatOpenAI(openai_api_key=api_key)  # The LLM instance used for summarization.
)

llm = ChatOpenAI(
    openai_api_key=api_key  # Initialize with the API key for authentication.
)
logging.debug("ChatOpenAI initialized with provided API key.")

# Step 4: Define system and human message templates for the prompt

# System message template: Provides instructions to the system
system_message_template = """You are an expert Python coder. Your task is to write a very short {language} function that will {task}. 
Please also provide a brief description of what the code does. Return the output as a JSON object with 'code' and 'description' fields."""
system_message_prompt_template = SystemMessagePromptTemplate.from_template(system_message_template)

# Human message template: Simulates human input (for example, specifying the task)
human_message_template = "{task}"
human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_message_template)

# Create a chat prompt template that combines both system and human message templates
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt_template, human_message_prompt_template]
)
logging.debug("Chat prompt template combining system and human messages defined.")

# Manual Chaining with Method Calls

try:
    # Step 5: Retrieve conversation history as messages
    history_messages = memory.load_memory_variables({'input': 'messages'})['messages']

    # Create the final prompt by combining history and the new prompt
    final_prompt = chat_prompt_template.format_prompt(language="python", task="return a list of numbers").to_messages()
    final_prompt = history_messages + final_prompt
    logging.debug("Final prompt created with system and human messages: %s", final_prompt)

    # Step 6: Invoke the LLM with the prompt
    llm_response = llm.invoke(final_prompt)
    logging.debug("LLM response received: %s", llm_response)

    # Save the response to memory
    memory.save_context({'input': final_prompt}, {'output': llm_response.content})

    # Since we removed the output parser, directly work with the LLM's response content
    response_content = llm_response.content
    logging.debug("Response content: %s", response_content)

    # Step 7: Generate a test for the generated code
    test_prompt = ChatPromptTemplate.from_template(
        "Write a unit test in python for the following function:\n\n{code}\n\n"
        "The test should validate the function's correctness."
    )
    test_prompt_text = test_prompt.format_prompt(code=response_content).to_messages()
    logging.debug("Formatted test prompt: %s", test_prompt_text)

    test_response = llm.invoke(test_prompt_text)
    logging.debug("Test response received: %s", test_response)

    # Save the test response to memory as well
    memory.save_context({'input': test_prompt_text}, {'output': test_response.content})

    # Step 8: Output the results
    print("Generated Code:")
    print(response_content)
    print("\nGenerated Test Code:")
    print(test_response.content.strip())
   
    # Step 9: Show the actual content of the conversation summary memory
    logging.debug("Retrieving conversation summary...")
    summary = memory.load_memory_variables({'input': 'messages'})['messages']
    
    if summary:
        print("\nConversation Summary Memory Content:")
        for msg in summary:
            print(f"{msg.type.capitalize()}: {msg.content}")
    else:
        print("No summary available or conversation did not produce a summary.")
except Exception as e:
    logging.error("An error occurred: %s", e)
    print(f"An error occurred: {e}")
