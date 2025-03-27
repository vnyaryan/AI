
# Importing the Chroma class from the langchain.vectorstores module for storing and searching vectors.
from langchain_community.vectorstores import Chroma

# Importing the OpenAIEmbeddings class from the langchain.embeddings module for generating embeddings.
from langchain_openai import OpenAIEmbeddings

# Importing the RetrievalQA class from the langchain.chains module for retrieval-based question answering.
from langchain.chains import RetrievalQA

# Importing the ChatOpenAI class from the langchain.chat_models module for interfacing with OpenAI's chat models.
from langchain_openai import ChatOpenAI


# Placeholder for an example API key for OpenAI services; in production, replace this with your actual API key and ensure it's stored securely.
api_key = "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"

# Instantiating a ChatOpenAI object to interface with OpenAI's chat models.
chat = ChatOpenAI(openai_api_key=api_key)

# Creating an instance of OpenAIEmbeddings using the provided API key. 
#This object will be used to generate embeddings.
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Instantiating a Chroma object for persisting embeddings to disk. 
#It uses the 'embeddings' object for generating embeddings.
db = Chroma(
    persist_directory="emb", # Directory where embeddings will be stored.
    embedding_function=embeddings # Function used to generate embeddings.
)

# Creating a retriever from the Chroma instance. 
#This retriever will be used in the retrieval-based QA chain.
retriever = db.as_retriever()

# Setting up a RetrievalQA chain with the specified components (chat model and retriever) and the chain type.
chain = RetrievalQA.from_chain_type(
    llm=chat, # Language (chat) model to use.
    retriever=retriever, # The retriever to use for fetching relevant context.
    chain_type="stuff" # Placeholder for the type of chain; replace "stuff" with the actual type as needed.
)

# Executing the chain with a query to retrieve an interesting fact about the English language.
result = chain.invoke("What is an interesting fact about the English language?")

# Printing the result of the retrieval-based question answering.
print(result)
