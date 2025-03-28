import os
import logging
from pinecone import Pinecone, ServerlessSpec
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings

# Constants
LOG_DIRECTORY = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\UDEMY_PROJECT-3\3'
LOG_FILENAME = 'az_cli-7.log'
API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"
PINECONE_API_KEY = "9fe3c832-5827-45f6-b066-e740b9b13e33"
PINECONE_ENVIRONMENT = "us-east-1"
TEST_INDEX1 = "testindex1"

# Setup logging
log_path = os.path.join(LOG_DIRECTORY, LOG_FILENAME)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_path, filemode='w')
logging.debug("Logging setup completed.")

# Initialize Pinecone
def initialize_pinecone(api_key, environment, index_name):
    """Initialize the Pinecone instance and retrieve the index."""
    try:
        pc = Pinecone(api_key=api_key)
        if index_name not in pc.list_indexes().names():
            logging.info(f"Index '{index_name}' not found. Creating a new index...")
            pc.create_index(
                name=index_name,
                dimension=1536,  # Update this dimension based on your embeddings
                metric='cosine',
                spec=ServerlessSpec(cloud="aws", region=environment)
            )
        index = pc.Index(index_name)  # Correct method to retrieve the index
        logging.debug(f"Pinecone index '{index_name}' initialized.")
        return index
    except Exception as e:
        logging.error(f"Failed to initialize Pinecone: {e}")
        raise

# Initialize Pinecone
try:
    index = initialize_pinecone(PINECONE_API_KEY, PINECONE_ENVIRONMENT, TEST_INDEX1)
except Exception as e:
    logging.error(f"Pinecone initialization failed: {e}")
    raise

# LangChain OpenAI client
llm = ChatOpenAI(api_key=API_KEY, model="gpt-3.5-turbo")

# OpenAI Embeddings client
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

# Query
query = "tell me about az login"
logging.debug(f"Query: {query}")

# Generate embedding
try:
    embedding = embeddings.embed_query(query)  # Correct method to generate embeddings
    logging.debug(f"Generated Embedding: {embedding}")
except Exception as e:
    logging.error(f"Error generating embedding: {e}")
    raise

# Perform similarity search
try:
    results = index.query(vector=embedding, top_k=5, include_metadata=True)
    logging.debug(f"Similarity search results: {results}")
except Exception as e:
    logging.error(f"Error performing similarity search: {e}")
    raise

# Build context
try:
    context = "\n".join([match.get('metadata', {}).get('text', '') for match in results.get('matches', [])])
    if not context.strip():
        raise ValueError("No context found in similarity search results.")
    logging.debug(f"Context for LLM: {context}")
except Exception as e:
    logging.error(f"Error building context: {e}")
    raise

# Generate response
try:
    prompt = f"You are an Azure CLI expert. Based on the following context, answer the user's question:\n{context}\n\nUser's question: {query}"
    response = llm.chat(messages=[{"role": "system", "content": prompt}])
    answer = response["choices"][0]["message"]["content"]
    logging.debug(f"LLM Response: {answer}")
except Exception as e:
    logging.error(f"Error generating response: {e}")
    raise

# Output response
print(answer)
