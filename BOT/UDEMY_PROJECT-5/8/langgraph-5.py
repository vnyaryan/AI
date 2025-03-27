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

# OpenAI API Key
OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Logging Setup
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\8'
        log_filename = 'langgraph-5.log'

        os.makedirs(log_directory, exist_ok=True)

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='w')
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")

setup_logging()

# Clear all_msg.txt file at the start of the script
all_msg_file_path = "all_msg.txt"
try:
    with open(all_msg_file_path, "w") as file:
        file.write("")  # Overwrite the file with no content
    logging.debug(f"Cleared content of {all_msg_file_path} at the start of the script.")
except Exception as e:
    logging.error(f"Failed to clear content of {all_msg_file_path}: {e}")

# Tool Definitions
@tool
def get_temperature(location: str):
    """
    Function to provide temperature details for a specified location.
    Example: "What's the temperature in San Francisco?"
    """
    if "sf" in location.lower() or "san francisco" in location.lower():
        return "It's 60 degrees and foggy."
    return "It's 100 degrees and sunny."

@tool
def get_population(country: str):
    """
    Function to provide the population of a location or country.
    Example: "What is the population of India?"
    """
    country = country.lower().strip()  # Normalize the input
    if "delhi" in country:
        return "Population of Delhi is 1 crore."
    return "Population is 5 crore."

tools = [get_temperature, get_population]
tool_node = ToolNode(tools)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY).bind_tools(tools)

# Global Tool Call Status
tool_call_status = False

# Node 1: Collect Input
def node1(state: MessagesState):
    user_message = input("You: ")
    if user_message.lower() == "exit":
        print("Conversation ended successfully.")
        exit(0)
    state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_message)])
    return state

# Node 2: Send Message to LLM
def node2(state: MessagesState):
    response = llm.invoke(state['messages'])
    return {"messages": [response]}

# Node 3: Process After Tool Execution
def node3(state: MessagesState):
    """
    Node 3: Handles execution after tools, if applicable.
    """
    try:
        logging.debug("NODE3 EXECUTION STARTED.")
        if tool_call_status:
            logging.debug("Tool call detected. Processing LLM response.")
            response = llm.invoke(state['messages'])
            logging.debug(f"LLM Response: {response}")
            return {"messages": [response]}
        logging.debug("No tool call detected. Skipping LLM response.")
        return {"messages": state['messages']}
    except Exception as e:
        logging.error(f"Error in NODE3 execution: {e}")
        raise

# Define Graph
builder = StateGraph(MessagesState)

# Add Nodes
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_node("tools", tool_node)
builder.add_node("node3", node3)

# Add Conditional Edges
builder.add_conditional_edges(
    "node2",
    lambda state: "tools" if state["messages"][-1].tool_calls else "node3"
)

builder.add_edge("tools", "node3")
builder.add_edge("node3", END)
builder.add_edge(START, "node1")
builder.add_edge("node1", "node2")

# Initialize Memory Saver
prompt_memory_saver = MemorySaver()

# Compile Graph with Interrupt
graph = builder.compile(
    checkpointer=prompt_memory_saver
)

# System Prompt
system_prompt = """
You are an intelligent assistant.

Your task is to decide whether to call functions `get_temperature` to know the temperature or `get_population` to know the population of a location or country based on the user query.

| **Keyword**      | **Function**       | **Description**                                  |
|-------------------|--------------------|-------------------------------------------------|
| "weather"         | get_temperature    | Provide temperature details for a location.     |
| "temperature"     | get_temperature    | Provide temperature details for a location.     |
| "population"      | get_population     | Provide population of a location or country.    |
"""

# Run Graph
if __name__ == "__main__":
    try:
        initial_state = {"messages": [SystemMessage(content=system_prompt)]}
        iteration = 0
        while True:
            iteration += 1
            tool_call_status = False
            updated_state = graph.invoke(initial_state, config={"configurable": {"thread_id": 1}})
            ai_response = updated_state["messages"][-1].content
            print(f"AI: {ai_response}")

            # Append interaction details to all_msg.txt
            with open("all_msg.txt", "a") as file:
                file.write("======== START OF ITERATION ========\n")
                file.write(json.dumps(
                    {"messages": [message.__dict__ for message in updated_state['messages']]},
                    indent=4
                ))
                file.write("\n======== END OF ITERATION ========\n\n")
            
            initial_state = updated_state
    except Exception as e:
        logging.error(f"Error during execution: {e}")
