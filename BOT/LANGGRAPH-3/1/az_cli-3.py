import os
import logging
from langchain_community.vectorstores import FAISS
from llama_cpp import Llama  # Using llama.cpp for Bling-Phi-3-GGUF embeddings

# Variables
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\LANGGRAPH-3\1\log'
log_filename = "az_cli-3.log"
faiss_index_path = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\LANGGRAPH-3\1\faiss_az_cli'
gguf_model_path = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\LANGGRAPH-3\1\bling-phi-3.gguf'  # Path to GGUF model

# Function to setup logging
def setup_logging():
    """Setup logging configuration with a fixed log filename."""
    try:
        os.makedirs(log_directory, exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(log_directory, log_filename),
            filemode='w'
        )
        logging.debug("✅ Logging setup completed.")
    except Exception as e:
        print(f"⚠️ Failed to setup logging: {e}")

# Load the GGUF-based embedding model
def initialize_embeddings():
    """Initialize the Bling-Phi-3 GGUF model using llama.cpp."""
    try:
        model = Llama(model_path=gguf_model_path, n_ctx=2048)  # Load the GGUF model
        logging.debug("✅ GGUF Model initialized successfully.")
        print("✅ GGUF Model initialized successfully.")
        return model
    except Exception as e:
        logging.error(f"❌ Failed to initialize GGUF model: {e}")
        raise

# Convert text to embedding using GGUF model
def get_embedding(model, text):
    """Generate embeddings for a given text using Bling-Phi-3-GGUF."""
    try:
        response = model(text, embedding=True)  # Generate embeddings
        return response["embedding"]
    except Exception as e:
        logging.error(f"❌ Failed to generate embeddings: {e}")
        raise

# Query FAISS Index to retrieve JSON filenames
def query_faiss_for_filenames(query, top_k=3):
    """Retrieve top-k matching JSON filenames from FAISS index for a given query."""
    try:
        # Load GGUF Model
        embedding_model = initialize_embeddings()
        vector_db = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)

        # Convert question into an embedding
        query_embedding = get_embedding(embedding_model, query)

        # Perform similarity search
        logging.info(f"🔍 Querying FAISS index for filenames: {query}")
        print(f"\n🔍 Querying FAISS index for filenames: {query}")

        docs = vector_db.similarity_search_by_vector(query_embedding, k=top_k)

        if not docs:
            logging.warning(f"⚠️ No matching filenames found for query: {query}")
            print("⚠️ No matching filenames found.")
            return []

        # Extract filenames from metadata
        filenames = [doc.metadata["source"] for doc in docs]

        # Log and display results
        logging.info(f"🔹 Found {len(filenames)} matching filenames.")
        print(f"\n🔹 Found {len(filenames)} matching filenames:")

        for idx, filename in enumerate(filenames, start=1):
            logging.info(f"📂 Match {idx}: {filename}")
            print(f"📂 Match {idx}: {filename}")

        return filenames

    except Exception as e:
        logging.error(f"❌ Error querying FAISS index: {e}")
        print(f"❌ Error querying FAISS index: {e}")
        return []

# Main function
def main():
    setup_logging()
    
    # Example query
    query = "How do I create an Azure virtual machine?"
    matched_filenames = query_faiss_for_filenames(query)

    # If needed, you can now fetch details from these JSON files

if __name__ == "__main__":
    main()
