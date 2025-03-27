# Import the ChatOpenAI class to utilize OpenAI's GPT models for generating chat responses.
from langchain_openai import ChatOpenAI

# Import the TextLoader class from the langchain.document_loaders module.
from langchain_community.document_loaders import TextLoader

# Sets an example API key; in real applications, this should be securely managed and not hardcoded.
# This is a placeholder for educational purposes only.
api_key = "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"

# Creating an instance of the ChatOpenAI class, initializing it with the API key for authentication.
# The verbose parameter is set to True to enable detailed logs, but it's currently commented out
# which means verbose logging is not enabled by default.
chat = ChatOpenAI(
    openai_api_key=api_key,  # This API key is used to authenticate requests to OpenAI's API.
   #verbose=True  # This line is commented out. If uncommented, it would set verbose logging to True.
)

# Create an instance of the TextLoader class, initializing it with the path to a text file named "facts.txt".
# This loader will be used to read and load the contents of the file.
loader = TextLoader("C:\\Users\\ARYAVIN\\Documents\\GitHub\\BOT\\UDEMY_PROJECT\\31\\facts\\facts.txt")

# Load the documents from the specified file using the loader's load method. 
# The contents of "facts.txt" are loaded into the 'docs' variable.
docs = loader.load()

# Print the documents loaded from "facts.txt" to the console.
# This is useful for debugging or verification purposes to ensure the content is loaded as expected.
print(docs)
