# Import necessary modules
import os
import logging
import json
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from typing import Annotated
from typing_extensions import TypedDict

# Step 1: Setup Logging
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\3'
        log_filename = 'langgraph-2.log'
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
llm_with_tools = llm.bind_tools([tool])

# Step 4: Define BasicToolNode
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# Step 5: Build the StateGraph
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

# Initialize the BasicToolNode and add it to the graph
tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# Define the routing function
def route_tools(state: State):
    """
    Route to the 'tools' node if the last message has tool calls.
    Otherwise, route to the end.
    """
    if messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

# Add conditional edges based on the routing function
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)

# Define the flow between nodes
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph
graph = graph_builder.compile()

# Step 6: Define Streaming Function
def stream_graph_updates(user_input: str):
    # Initialize the state
    state = {"messages": [("user", user_input)]}
    logging.debug(f"Streaming state: {state}")
    for event in graph.stream(state):
        for value in event.values():
            assistant_message = value["messages"][-1].content if value["messages"] else "No content"
            logging.debug(f"Assistant message: {assistant_message}")
            print("Assistant:", assistant_message)

# Step 7: Interactive Conversation Loop
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
