import sys
import os
from llama_cpp import Llama

def safe_print(data):
    print(str(data).encode('utf-8', errors='ignore').decode('utf-8'))

# Define the model path (Linux style)
model_path = "/tmp/zen"  #  Your model path

# Construct the full path to the .gguf file
# Assuming you downloaded the Q4_K_S version, adjust if needed
gguf_file = os.path.join(model_path, "bling-phi-3.gguf")  # Adjust if necessary

# Check if the gguf file exists
if not os.path.exists(gguf_file):
    print(f"Error: GGUF file not found at {gguf_file}")
    exit()

try:
    # Initialize the Llama model
    llm = Llama(
        model_path=gguf_file,
        n_gpu_layers=0,  # Important: Specify 0 to not use GPU
        n_ctx=2048,  # Adjust based on your needs and available RAM
        n_threads=4,  # Adjust based on your CPU cores
        verbose=False,
    )

    # Create a prompt
    prompt = "What is the capital of France?"

    # Run inference
    output = llm(
        prompt,
        max_tokens=150,  # Adjust as needed
        stop=["<|file_separator|>"],  # Use the appropriate Stop Sequence for this model
        temperature=0.7,  # Adjust for creativity
    )

    # Print the output
    safe_print(output["choices"][0]["text"])

except Exception as e:
    print(f"Error loading or running model: {e}")
