import os
import logging
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Literal
import json
import openai
from langsmith import wrappers, traceable
from pydantic import BaseModel, Field
from langsmith import Client

# Logging Setup
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\UDEMY_PROJECT-6\2'
        log_filename = 'langsmith-2.log'
        os.makedirs(log_directory, exist_ok=True)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='w')
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")

# Initialize logging
setup_logging()

# OpenAI API Key and LangSmith API Key
OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"
LANGCHAIN_API_KEY = "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"

# Set environment variables for API keys
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# Set API key for OpenAI client
openai.api_key = OPENAI_API_KEY

# Create an instance of the LangSmith Client directly
client = Client()

# Step 3: Create a Dataset in LangSmith
examples = [
    ("Which is  polluted country in world ?", "delhi."),
    ("What is capital of  ?", "Earth's lowest point is The Dead Sea."),
]

inputs = [{"question": input_prompt} for input_prompt, _ in examples]
outputs = [{"answer": output_answer} for _, output_answer in examples]

# Create a new dataset in LangSmith
dataset = client.create_dataset(
    dataset_name="Sample dataset", 
    description="A sample dataset in LangSmith."
)

# Add examples to the dataset
client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)

# Step 4: Define the Target Function for Evaluation
def target(inputs: dict) -> dict:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Answer the following question accurately"},
            {"role": "user", "content": inputs["question"]},
        ],
    )
    return {"response": response.choices[0].message.content.strip()}

# Step 5: Define the Evaluator Function to Assess Accuracy of Responses
class Grade(BaseModel):
    score: bool = Field(description="Indicates whether the response is accurate.")

instructions = """Evaluate Student Answer against Ground Truth for conceptual similarity and classify true or false:
- False: No conceptual match and similarity.
- True: Most or full conceptual match and similarity."""

def accuracy(outputs: dict, reference_outputs: dict) -> bool:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"Ground Truth answer: {reference_outputs['answer']}; Student's Answer: {outputs['response']}"},
        ],
    )
    return response.choices[0].message.content.strip().lower() == 'true'

# Step 6: Run Evaluation and View Results in LangSmith Dashboard
experiment_results = client.evaluate(
    target,
    data="Sample dataset",
    evaluators=[accuracy],
    experiment_prefix="first-eval-in-langsmith",
    max_concurrency=2,
)

# Print evaluation results to console for review
print("Evaluation Results:")
for result in experiment_results:
    print(result)
