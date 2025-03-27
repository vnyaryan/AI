"""
This script performs the following tasks:

1. **Logging Setup**: 
   - Configures logging to a specific file located in the specified directory.
   - Ensures the log directory exists and writes logs in append mode.

2. **API Key Management and LLM Initialization**:
   - Securely manages the OpenAI API key and initializes the `ChatOpenAI` class using this key.

3. **Pydantic Model Definition**:
   - Defines a Pydantic model `CodeOutput` to structure the output from the language model. 
   The model includes fields for the generated code snippet and a description.

4. **Output Parser Initialization**:
   - Initializes an output parser using the Pydantic model to structure the language model's output.

5. **Prompt Template Definition**:
   - Defines system and human message templates for creating prompts. 
   The system template guides the language model on what code to generate, and the human template simulates task input.

6. **Prompt Creation and LLM Invocation**:
   - Combines the system and human message templates into a final prompt.
   - Invokes the language model with the prompt and logs the response.

7. **Response Parsing**:
   - Parses the language model's response into a structured format using the previously defined Pydantic model.

8. **Test Code Generation**:
   - Generates a unit test for the generated code by creating and invoking a new prompt.

9. **Output Display**:
   - Prints the generated code, its description, and the generated unit test code.

10. **Error Handling**:
    - Catches and logs any errors encountered during the process, ensuring graceful termination of the script in case of issues.
"""

import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

def setup_logging():
    """Setup logging configuration with a fixed log filename."""
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT\22'
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
llm = ChatOpenAI(
    openai_api_key=api_key  # Initialize with the API key for authentication.
)
logging.debug("ChatOpenAI initialized with provided API key.")

# Step 3: Define a Pydantic model for the code output
class CodeOutput(BaseModel):
    code: str = Field(..., description="The generated code snippet")
    description: str = Field(..., description="A brief description of what the code does")
logging.debug("Pydantic model for code output defined.")

# Step 4: Define the OutputParser using the Pydantic model
output_parser = PydanticOutputParser(pydantic_object=CodeOutput)
logging.debug("Output parser initialized.")

# Step 5: Define system and human message templates for the prompt

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
    # Step 6: Create the prompt
    final_prompt = chat_prompt_template.format_prompt(language="python", task="return a list of numbers").to_messages()
    logging.debug("Final prompt created with system and human messages: %s", final_prompt)

    # Step 7: Invoke the LLM with the prompt
    llm_response = llm.invoke(final_prompt)
    logging.debug("LLM response received: %s", llm_response)

    # Step 8: Parse the LLM response
    parsed_response = output_parser.parse(llm_response.content)
    logging.debug("Parsed response: Code - %s, Description - %s", parsed_response.code, parsed_response.description)

    # Step 9: Generate a test for the generated code
    test_prompt = ChatPromptTemplate.from_template(
        "Write a unit test in python for the following function:\n\n{code}\n\n"
        "The test should validate the function's correctness."
    )
    test_prompt_text = test_prompt.format_prompt(code=parsed_response.code)
    logging.debug("Formatted test prompt: %s", test_prompt_text)

    test_response = llm.invoke(test_prompt_text)
    logging.debug("Test response received: %s", test_response)

    # Step 10: Output the results
    print("Generated Code:")
    print(parsed_response.code)
    print("\nDescription:")
    print(parsed_response.description)
    print("\nGenerated Test Code:")
    print(test_response.content.strip())

except Exception as e:
    logging.error("An error occurred: %s", e)
    print(f"An error occurred: {e}")
