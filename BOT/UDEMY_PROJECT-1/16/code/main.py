import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

def setup_logging():
    """Setup logging configuration with a fixed log filename."""
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT\16'
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

# Step 5: Define a prompt template for generating code based on input variables (programming language and task).
code_prompt = ChatPromptTemplate.from_template(
    "Write a very short {language} function that will {task}. "
    "Also provide a brief description of what the code does."
    "Return the output as a JSON object with 'code' and 'description' fields."
)
logging.debug("Prompt template for code generation defined.")

# Manual Chaining with Method Calls

try:
    # Step 6: Create the prompt
    prompt_text = code_prompt.format_prompt(language="python", task="return a list of numbers")
    logging.debug("Formatted prompt: %s", prompt_text)

    # Step 7: Invoke the LLM with the prompt
    llm_response = llm.invoke(prompt_text)
    logging.debug("LLM response received: %s", llm_response)

    # Step 8: Parse the LLM response
    parsed_response = output_parser.parse(llm_response.content)
    logging.debug("Parsed response: Code - %s, Description - %s", parsed_response.code, parsed_response.description)

    # Step 9: Generate a test for the generated code
    test_prompt = ChatPromptTemplate.from_template(
        "Write a unit test in python for the following function:\n\n{code}\n\n"
        "The test should validate the function's correctness."
    )
    test_prompt_text = test_prompt.format_prompt(language="python", code=parsed_response.code)
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
