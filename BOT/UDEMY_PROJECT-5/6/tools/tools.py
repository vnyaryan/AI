from langchain_core.tools import tool
import os

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
        output_folder = r"C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\6\tools"
        output_filename = "pythonscript1.py"
        output_path = os.path.join(output_folder, output_filename)

        # Create the folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Template for the generated Python script
        script_template = f'''"""
This script is auto-generated based on the following instruction:
{instruction}
"""

# Add your imports here
import sys

def main():
    """
    Main function to implement the logic based on the instruction.
    """
    print("Instruction: {instruction}")
    print("This is a placeholder for your logic.")

if __name__ == "__main__":
    main()
'''
        # Save the generated script to the specified file
        with open(output_path, 'w') as script_file:
            script_file.write(script_template)

        return f"Python script created successfully: {output_path}"
    except Exception as e:
        return f"Error creating the Python script: {e}"
