"""
This script performs the following tasks:

1. **Logging Setup**: 
   - Configures logging to a specific file located at `/tmp/certificate/log/certificate_verification.log`.
   - Ensures the log directory exists and writes logs in append mode.

2. **Language Model Interaction**:
   - Uses OpenAI's language model to generate code snippets based on provided input parameters (programming language and task).
   - Parses the output of the language model into a structured format using Pydantic models.

3. **Prompt Definition**:
   - Defines a prompt template to generate a short function in a specified programming language that performs a specific task.
   - Also requests a brief description of what the generated function does.

4. **Response Handling**:
   - The response from the language model is parsed into a structured format, including the generated code and a description.
   - Logs the generated code and description for further analysis or use.

5. **Error Handling**:
   - Logs any errors encountered during prompt creation, model invocation, or output parsing.
   - Exits the script gracefully if any errors occur.
"""

import os
import logging
from langchain_openai import ChatOpenAI  # Use ChatOpenAI for chat-oriented model interactions.
from langchain_core.prompts import ChatPromptTemplate  # For creating structured prompts.
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

import os
import logging

def setup_logging():
    """Setup logging configuration with a fixed log filename."""
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT\15'
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

# Step 3: Define a Pydantic model for the output
class CodeOutput(BaseModel):
    code: str = Field(..., description="The generated code snippet")
    description: str = Field(..., description="A brief description of what the code does")
logging.debug("Pydantic model for output defined.")

# Step 4: Define the OutputParser using the Pydantic model
output_parser = PydanticOutputParser(pydantic_object=CodeOutput)
logging.debug("Output parser initialized.")

# Step 5: Define a prompt template for generating code based on input variables (programming language and task).
code_prompt = ChatPromptTemplate.from_template(
    "Write a very short {language} function that will {task}. "
    "Also provide a brief description of what the code does."
)
logging.debug("Prompt template defined.")

# Step 6: Chain the prompt template, model, and output parser together using the pipe operator.
code_chain = code_prompt | llm | output_parser
logging.debug("Code generation chain created.")

# Step 7: Generate a code snippet by invoking the chain with specific inputs (language and task).
response = code_chain.invoke({  # Using invoke instead of __call__ for LangChain >= 0.1.0.
    "language": "python",  # Programming language to generate code in.
    "task": "return a list of numbers"  # Task description for the code.
})
logging.debug("Code generation invoked with inputs: language='python', task='return a list of numbers'.")

# Step 8: Extract and print the structured output.
logging.debug("Generated code: %s", response.code)
logging.debug("Description: %s", response.description)

print("Generated Code:")
print(response.code)  # Extract and print the generated code.

print("\nDescription:")
print(response.description)  # Extract and print the description of the code.
