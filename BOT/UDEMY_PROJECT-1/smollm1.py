# Step 1: Import necessary modules from LangChain for interacting with Hugging Face's SmolLM2 model.
from langchain_huggingface import HuggingFaceEndpoint  # Recommended import for Hugging Face endpoints.
from langchain.prompts import PromptTemplate  # For creating structured prompts.
from langchain.chains import LLMChain  # For chaining prompts and model responses.

# Step 2: Securely manage your Hugging Face API token and set the endpoint URL.
huggingfacehub_api_token = "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"  # Provided API token.
# Set the endpoint URL to the SmolLM2-135M repository.
endpoint_url = "https://api-inference.huggingface.co/models/HuggingFaceTB/SmolLM2-360M"

llm = HuggingFaceEndpoint(
    endpoint_url=endpoint_url,
    huggingfacehub_api_token=huggingfacehub_api_token,
    temperature=0.7  # Pass temperature explicitly as a parameter.
)

# Step 3: Define a prompt template for generating code based on input variables.
code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

# Step 4: Chain the prompt template and the model together using the pipe operator.
code_chain = code_prompt | llm

# Step 5: Generate a code snippet by invoking the chain with specific inputs.
result = code_chain.invoke({
    "language": "python",             # The programming language for the generated code.
    "task": "return a list of numbers"  # The task description for the code.
})

# Step 6: Print the generated code snippet to the console.
print(result)
