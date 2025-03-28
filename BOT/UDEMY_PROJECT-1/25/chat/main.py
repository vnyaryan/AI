# Import the ChatOpenAI class to utilize OpenAI's GPT models for generating chat responses.
from langchain_openai import ChatOpenAI
# Import the LLMChain class for creating a sequence of operations involving language models.
from langchain.chains import LLMChain
# Import necessary components for structuring the chat prompts and managing placeholders.
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
# Import classes for in-memory conversation management and file-based message history.
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

# Security best practices reminder.
# Sets an example API key; in real applications, this should be securely managed and not hardcoded.
api_key = "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"

# Creating an instance of the ChatOpenAI class, initializing it with the API key for authentication.
chat = ChatOpenAI(
    openai_api_key=api_key  # This API key is used to authenticate requests to OpenAI's API.
)

# Initialize a ConversationBufferMemory for conversation context management.
# It uses a FileChatMessageHistory to store and retrieve conversation history from a JSON file.
memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),  # Specifies the file for storing message history.
    memory_key="messages",  # The key under which messages are stored and retrieved in conversation context.
    return_messages=True  # Configures the memory to include past messages in the response for continuity.
)

# Set up a chat prompt template incorporating both the incoming content and the conversation history.
prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],  # Defines the variables to be used in the prompt template.
    messages=[
        MessagesPlaceholder(variable_name="messages"),  # Placeholder for injecting past messages into the prompt.
        HumanMessagePromptTemplate.from_template("{content}")  # Template for the current user input.
    ]
)

# Create an LLMChain instance to manage the flow of processing input, using the chat model, prompt, and memory setup.
chain = LLMChain(
    llm=chat,  # Specifies the ChatOpenAI instance for response generation.
    prompt=prompt,  # The structured prompt template for interaction.
    memory=memory  # The conversation memory management setup, incorporating file-based history.
)

# Enter an infinite loop for interactive chat sessions.
while True:
    content = input(">> ")  # Capture user input from the console.

    # Generate a response by processing the current input along with the conversation context through the chain.
    result = chain.invoke({"content": content})  # The input content is passed to the chain for processing.

    print(result["text"])  # Print the generated response to the console.
