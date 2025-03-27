

# Import the OpenAIEmbeddings class from langchain
from langchain_openai.embeddings import OpenAIEmbeddings

# Define the API key directly in the code
# WARNING: For real applications, especially those shared or deployed, consider securing the API key outside the codebase.
api_key = "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"

try:
    # Create an instance of the OpenAIEmbeddings class with the API key
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Generate embeddings for a query
    emb = embeddings.embed_query("hi")

    # Placeholder for handling the embedding output
    # Here you would add your logic to use 'emb', such as analyzing it, or integrating it with other components
    print(emb)  # Example action: print the embedding output

except Exception as e:
    # Basic error handling to catch and report exceptions during the embedding process
    print(f"An error occurred: {e}")

# Reminder: Directly including sensitive information like an API key in your code is not advisable for production environments.
