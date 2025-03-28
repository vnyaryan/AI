import streamlit as st  # Importing Streamlit for creating the web app interface
from langchain_openai import ChatOpenAI # Importing OpenAI model from LangChain to interact with OpenAI's API

# Define the API key here (predefined, no need for user input)
API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Set the title of the Streamlit app (visible at the top of the interface)
st.title("🦜🔗 Langchain Quickstart App")

# Function to generate a response using the OpenAI model
def generate_response(input_text):
    """
    This function takes user input (text prompt) and generates a response using the OpenAI model.
    
    Input:
    - input_text (str): The prompt text entered by the user.

    Output:
    - Displays the generated response using the Streamlit info box.
    """
    # Initializing the OpenAI model with the API key and a temperature of 0.7 (controls creativity of responses)
    llm = ChatOpenAI(openai_api_key=API_KEY)
    
    # Generate and display the response based on the user's input
    st.info(llm(input_text))  # Displays the response in an info box in the Streamlit app

# Create a form for user input (This form is shown on the main page of the app)
with st.form("my_form"):
    """
    This form allows the user to input text that will be sent to the OpenAI model.
    
    Input:
    - User's text prompt (entered via text area).
    
    When the user clicks "Submit", the form is submitted and the prompt is processed.
    """
    # Text area for the user to input their prompt. Default prompt is provided.
    text = st.text_area("Enter text:", "What are 3 key advice for learning how to code?")
    
    # Submit button for the form
    submitted = st.form_submit_button("Submit")

    # Check if the API key is available
    if not API_KEY:
        # If the API key is missing, display a message to inform the user
        st.info("API key is missing!")
    
    # If the form is submitted and the API key is valid, generate the response
    elif submitted:
        # Call the function to generate a response based on the user's input
        generate_response(text)
