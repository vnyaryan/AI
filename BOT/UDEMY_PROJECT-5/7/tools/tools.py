from langchain_core.tools import tool
import os
from langchain_openai import ChatOpenAI

OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

@tool
def create_python_script(instruction: str) -> str:
    """
    Tool to generate a Python script based on the provided instruction and save it to a specific location.

    Args:
        instruction (str): The user's instruction for what the script should do.

    Returns:
        str: A success message with the script's filename and location.
    """
    try:
        # Define the output folder and file name
        output_folder = r"C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\7\tools"
        output_filename = "pythonscript1.py"
        output_path = os.path.join(output_folder, output_filename)

        # Create the folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Use OpenAI to generate Python logic
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        prompt = f"""
        Create a Python script for the following instruction:
        {instruction}
        
        The script should include:
        - Necessary imports
        - Function definitions
        - Main logic implementation
        - Comments explaining the code
        """
        response = llm.invoke([instruction]).content.strip()

        # Save the generated script to the specified file
        with open(output_path, 'w') as script_file:
            script_file.write(response)

        return f"Python script created successfully: {output_path}"
    except Exception as e:
        return f"Error creating the Python script: {e}"
