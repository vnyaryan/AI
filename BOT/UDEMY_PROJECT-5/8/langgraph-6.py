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
        log_filename = 'langgraph-6.log'

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
    """
    if "sf" in location.lower() or "san francisco" in location.lower():
        return "It's 60 degrees and foggy."
    return "It's 100 degrees and sunny."

@tool
def get_population(country: str):
    """
    Function to provide the population of a location or country.
    """
    country = country.lower().strip()
    if "delhi" in country:
        return "Population of Delhi is 1 crore."
    return "Population is 5 crore."

tools = [get_temperature, get_population]
tool_node = ToolNode(tools)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY).bind_tools(tools)

# Node 1: Collect Input

def node1(state: MessagesState):
    """
    Node 1: Collects a single user input and adds it to the state.
    """
    try:
        logging.debug("NODE1 EXECUTION STARTED.")
        
        # Prompt the user for input
        user_message = input("You: ")

        # Check for termination condition
        if user_message.lower() == "exit":
            logging.debug("Termination condition met in Node 1. Exiting conversation.")
            print("Conversation ended successfully.")
            logging.debug("NODE1 EXECUTION COMPLETED.")
            exit(0)  

        # Add the user message to the state using add_messages
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_message)]) 

        # Log details
        last_message = state["messages"][-1]
        logging.debug(f"User message added to state: {last_message.content}")
        logging.debug("NODE1 EXECUTION COMPLETED.")
        return state
    except Exception as e:
        logging.error(f"Error in node_1: {e}")
        raise

# Node 2: Send Message to LLM
def node2(state: MessagesState):
    """
    Node 2: SEND MESSAGE TO LLM 
    """    
    messages = state['messages']
    try:
        logging.debug("NODE2 EXECUTION STARTED.")
        response = llm.invoke(messages)
        logging.info("Model invoked successfully.")
        logging.debug(f"Model response: {response}")
        logging.debug("NODE2 EXECUTION COMPLETED.")        
        return {"messages": [response]} 
    except Exception as e:
        logging.error(f"Error invoking model: {e}")
        raise

# HITL: Ask Human for Approval
def ask_human_review(tool_calls):
    if not tool_calls:
        return "node3"  # Skip tool execution

    print("Tool Calls Detected:")
    for i, tool_call in enumerate(tool_calls, 1):
        print(f"{i}. Tool: {tool_call['name']}, Args: {tool_call['args']}")

    print("Options: ")
    print("1. Approve tool call")
    print("2. Skip tool execution")
    print("3. Exit")

    user_choice = input("Enter your choice (1/2/3): ")

    if user_choice == "1":
        print("Tool execution approved.")
        return "tools"  # Proceed to tool execution
    elif user_choice == "2":
        print("Tool execution skipped.")
        return "node3"  # Skip execution
    elif user_choice == "3":
        print("Exiting the application.")
        exit(0)
    else:
        print("Invalid choice. Skipping tool execution.")
        return "node3"

# Node 3: Process After Tool Execution

def node3(state: MessagesState):
    """
    Node 3: Invokes the LLM only if the global tool_call_status is True.
    """

    try:
        logging.debug("NODE3 EXECUTION STARTED.")
        
        if state.get("tool_call_status", False):
            logging.debug("No tool call detected. Skipping LLM invocation.")
            logging.debug("NODE3 EXECUTION COMPLETED.")
            return {"messages": state['messages']}  # Return unchanged state
        else:
            # Invoke the LLM if a tool call is detected
            logging.debug("Tool call detected. Invoking LLM.")
            response = llm.invoke(state['messages'])
            logging.info("LLM invoked successfully.")
            logging.debug(f"LLM response: {response}")
            logging.debug("NODE3 EXECUTION COMPLETED.")        
            return {"messages": [response]} 
    except Exception as e:
        logging.error(f"Error in NODE3: {e}")
        raise

# Define Graph
builder = StateGraph(MessagesState)

# Add Nodes
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_node("tools", tool_node)
builder.add_node("node3", node3)


#Add Edges
builder.add_edge(START, "node1")   # Start the flow at node1
builder.add_edge("node1", "node2")  # Move to node2 after collecting input
builder.add_conditional_edges("node2", lambda state: ask_human_review(state["messages"][-1].tool_calls)) # Optional: Connect node2 to tools
builder.add_edge("tools", "node3")  # Move to node3 after tool execution
builder.add_edge("node3", END)      # Terminate the flow


# Initialize Memory Saver
prompt_memory_saver = MemorySaver()

# Compile Graph with Interrupt
graph = builder.compile(
    checkpointer=prompt_memory_saver
)
# Function to write MessagesState details to a text file
def write_all_message_types_to_txt(state_values, output_file="C:\\Users\\ARYAVIN\\Documents\\GitHub\\BOT\\UDEMY_PROJECT-5\\8\\msg_details.txt"):
    """
    Processes and writes the content of all message types (HumanMessage, AIMessage, ToolMessage, etc.)
    to a text file with separators.
    """
    try:
        with open(output_file, "w") as file:
            file.write("======== START OF MESSAGES ========\n\n")
            for message_group in state_values:
                for message in message_group:
                    if isinstance(message, HumanMessage):
                        file.write(f"HumanMessage Content: {message.content}\n")
                        file.write("-------------------------------------------------------------------\n")
                    elif isinstance(message, AIMessage):
                        file.write(f"AIMessage Content: {message.content}\n")
                        file.write("-------------------------------------------------------------------\n")
                    elif isinstance(message, SystemMessage):
                        file.write(f"SystemMessage Content: {message.content}\n")
                        file.write("-------------------------------------------------------------------\n") 
                    elif isinstance(message, ToolMessage):
                        file.write(f"ToolMessage Content: {message.content}\n")
                        file.write(f"  Tool Name: {message.name}\n")
                        file.write("-------------------------------------------------------------------\n")
            file.write("\n======== END OF MESSAGES ========\n")
        logging.debug(f"All message types successfully written to {output_file}")
    except Exception as e:
        logging.error(f"Error writing message types to file: {e}")

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

# Invoke Graph
if __name__ == "__main__":
    try:
        logging.debug("Starting multi-turn conversation.")
        initial_state = {"messages": [SystemMessage(content=system_prompt)], "tool_call_status": False}
        iteration = 0

        while True:
            iteration += 1
            logging.debug(f"START ITERATION - {iteration}")

            # Invoke the graph
            updated_state = graph.invoke(initial_state, config={"configurable": {"thread_id": 1}})
            ai_response = updated_state["messages"][-1].content
            print(f"AI: {ai_response}")

            # Append interaction details to all_msg.txt
            with open("all_msg.txt", "a") as file:
                file.write("======== START OF ITERATION ========\n")
                file.write(json.dumps(
                    {"messages": [message.__dict__ for message in updated_state["messages"]]},
                    indent=4
                ))
                file.write("\n======== END OF ITERATION ========\n\n")

            initial_state = updated_state
            
            # Write the messages to the specified file
            write_all_message_types_to_txt(updated_state.values())
            logging.debug(f"ENDING ITERATION - {iteration}")
            
    except Exception as e:
        logging.error(f"Error during multi-turn conversation: {e}")
        raise
