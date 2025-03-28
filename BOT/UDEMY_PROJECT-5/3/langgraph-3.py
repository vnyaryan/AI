# Import necessary modules
import os
import logging
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict

# Step 1: Setup Logging
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\3'
        log_filename = 'langgraph-3.log'
        os.makedirs(log_directory, exist_ok=True)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='w')
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")

setup_logging()

# Step 2: Define the State Schema
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Step 3: Initialize OpenAI API and Tavily Tool
# Use provided API keys
OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"
TAVILY_API_KEY = "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"

# Set the Tavily API key as an environment variable
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Initialize OpenAI's Chat model
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# Initialize TavilySearchResults tool
tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True
)

# Create a list of tools
tools = [tool]

# Bind the tool to the language model
llm_with_tools = llm.bind_tools(tools)

# Step 4: Build the StateGraph
graph_builder = StateGraph(State)

# Define the chatbot function to process state and use tools if needed
def chatbot(state: State):
    # Log input state for debugging
    logging.debug(f"Input state: {state}")
    response = llm_with_tools.invoke(state["messages"])
    logging.debug(f"Response from llm_with_tools: {response}")
    return {"messages": [response]}

# Add chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Initialize the ToolNode and add it to the graph
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add conditional edges based on the tools_condition
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Define the flow between nodes
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph
graph = graph_builder.compile()

# Step 5: Define Streaming Function
def stream_graph_updates(user_input: str):
    # Initialize the state
    state = {"messages": [("user", user_input)]}
    logging.debug(f"Streaming state: {state}")
    for event in graph.stream(state):
        for value in event.values():
            assistant_message = value["messages"][-1].content if value["messages"] else "No content"
            logging.debug(f"Assistant message: {assistant_message}")
            print("Assistant:", assistant_message)

# Step 6: Interactive Conversation Loop
if __name__ == "__main__":
    print("Welcome to the LangGraph Chatbot! Type 'quit' to exit the conversation.")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            print("An unexpected error occurred. Ending conversation.")
            break
