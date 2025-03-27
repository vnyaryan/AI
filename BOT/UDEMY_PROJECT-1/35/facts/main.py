# Import the ChatOpenAI class to utilize OpenAI's GPT models for generating chat responses.
from langchain_openai import ChatOpenAI

# Import the CharacterTextSplitter class for splitting text into chunks based on character count.
from langchain.text_splitter import CharacterTextSplitter

# Import the TextLoader class from the langchain.document_loaders module for loading documents.
from langchain_community.document_loaders import TextLoader

# Sets an example API key; in real applications, this should be securely managed and not hardcoded.
# This is a placeholder for educational purposes only.
api_key = "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"

# Creating an instance of the ChatOpenAI class, initializing it with the API key for authentication.
# The verbose parameter is commented out, implying verbose logging is not enabled by default.
chat = ChatOpenAI(
    openai_api_key=api_key,  # This API key is used to authenticate requests to OpenAI's API.
    #verbose=True  # This line is commented out. If uncommented, it would set verbose logging to True.
)

# Creating an instance of the CharacterTextSplitter class.
# It's initialized to split text at every newline character, with each chunk having up to 200 characters without any overlap.
text_splitter = CharacterTextSplitter(
    separator="\n",  # Defines the character used to split the text.
    chunk_size=100,  # The maximum number of characters in each text chunk.
    chunk_overlap=0  # The number of characters to overlap between consecutive chunks (set to 0 for no overlap).
)

# Create an instance of the TextLoader class, initializing it with the full path to a text file named "facts.txt".
# This loader will be used to read and load the contents of the file.
loader = TextLoader("C:\\Users\\ARYAVIN\\Documents\\GitHub\\BOT\\UDEMY_PROJECT\\35\\facts\\facts.txt")

# Using the 'load_and_split' method of the loader to load the document and then split it into chunks based on the specified text splitter.
# This method is particularly useful for large documents or when you need to process the text in smaller, manageable pieces.
docs = loader.load_and_split(
    text_splitter=text_splitter  # Passing the previously defined text splitter to the load_and_split method.
)

# Iterating over each chunked document generated by the text splitter.
for doc in docs:
    print(doc.page_content)  # Printing the content of each chunk to the console.
    print("\n")  # Adding a newline for better readability between chunks.
