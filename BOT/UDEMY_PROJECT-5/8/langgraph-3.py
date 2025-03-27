import os
import logging
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage,SystemMessage, ChatMessage, FunctionMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Literal
import json


OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Step 1: Setup Logging
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\8'
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

# Step 5: Define the Tool Node
@tool
def get_temperature(location: str):
    """
    Function to provide temperature details for a specified location.
    Example: "What's the temperature in San Francisco?"
    """
    if "sf" in location.lower() or "san francisco" in location.lower():
        return "It's 60 degrees and foggy."
    return "It's 100 degrees and sunny."

def get_population(country: str):
    """
    Function to provide the population of  any location
    Example: "What is population of  noida?"
    """
    country = country.lower().strip()  # Normalize the input

    if "india" in country:
        return "population of Delhi is 1cr"
    else:
        return "poupulation is 5 crore"


tools = [get_temperature,get_population]

tool_node = ToolNode(tools)

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY).bind_tools(tools)

# Global variable to track tool call status
tool_call_status = False

# Step 3: Define Node 1 - Collect One Input
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
            exit(0)  # Exit the script

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

# Define the function that calls the model
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

def node3(state: MessagesState):
    """
    Node 3: Invokes the LLM only if the global tool_call_status is True.
    """
    global tool_call_status
    try:
        logging.debug("NODE3 EXECUTION STARTED.")
        
        if not tool_call_status:
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

# Define the function that determines whether to continue or not
# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", "node3"]:
    global tool_call_status
    logging.debug("ROUTING FUNCTION STARTED.")
    messages = state['messages']
    last_message = messages[-1]

    try:
        # Check if the last message contains a tool call
        if last_message.tool_calls:
            tool_call = last_message.tool_calls[0]  # Assume only one tool call at a time
            function_name = tool_call["name"]  # Extract function name
            
            # Log the detected tool call and function name
            logging.debug(f"Tool call detected: {tool_call}")
            logging.debug(f"Function name in tool call: {function_name}")
            
            # Check if the function name matches a registered tool
            if function_name == "get_temperature":
                logging.debug("Routing to TOOL NODE for get_temperature.")
                tool_call_status = True
                return "tools"
            elif function_name == "get_population":
                logging.debug("Routing to TOOL NODE for get_population.")
                tool_call_status = True
                return "tools"
            else:
                # Handle unexpected or unregistered tools
                logging.warning(f"Unrecognized function '{function_name}' in tool call.")
                tool_call_status = False
                return "node3"

        # Default routing to NODE3 if no tool call is detected
        logging.debug("No tool call detected. Routing to NODE3.")
        tool_call_status = False
        return "node3"

    except Exception as e:
        logging.error(f"Error in routing logic: {e}")
        # Fallback to NODE3 in case of an error
        return "node3"


# Define a new graph
builder = StateGraph(MessagesState)

# Add Nodes
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_node("tools", tool_node)
builder.add_node("node3", node3)

# Add edges
builder.add_edge(START, "node1")
builder.add_edge("node1", "node2")

builder.add_conditional_edges("node2", should_continue)
builder.add_edge("tools", "node3")
builder.add_edge("node3", END)

# Initialize Memory Saver
prompt_memory_saver = MemorySaver()

# Compile Graph
graph = builder.compile(checkpointer=prompt_memory_saver)

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
                    elif isinstance(message, ChatMessage):
                        file.write(f"ChatMessage Content: {message.content}\n")
                        file.write("-------------------------------------------------------------------\n")
                    elif isinstance(message, FunctionMessage):
                        file.write(f"FunctionMessage Content: {message.content}\n")
                        file.write(f"  Function Name: {message.name}\n")
                        file.write("-------------------------------------------------------------------\n")
                    elif isinstance(message, ToolMessage):
                        file.write(f"ToolMessage Content: {message.content}\n")
                        file.write(f"  Tool Name: {message.name}\n")
                        file.write("-------------------------------------------------------------------\n")
            file.write("\n======== END OF MESSAGES ========\n")
        logging.debug(f"All message types successfully written to {output_file}")
    except Exception as e:
        logging.error(f"Error writing message types to file: {e}")

# Define the system prompt
system_prompt = """
You are an intelligent assistant.

Your task is to decide whether to call functions `get_temperature` to know the temperature or weather details or `get_population` to know the current  population of any location or any country based on user query.

Use the following table as your guide:

| **Keyword**      | **Function**       | **Description**                                  |
|-------------------|--------------------|-------------------------------------------------|
| "weather"         | get_temperature    | Provide temperature details for a location.     |
| "temperature"     | get_temperature    | Provide temperature details for a location.     |
| "population"      | get_population     | Provide population of country or any location   |

If the user query matches a keyword, decide which function to call and prepare the arguments for that function. If no keyword matches, respond directly to the user with an appropriate message.
"""


# Step 9: Invoke Graph
if __name__ == "__main__":
    try:
        logging.debug("Starting multi-turn conversation.")
        #initial_state = {"messages": []}
        initial_state = {"messages": [SystemMessage(content=system_prompt)]}
        iteration = 0

        while True:
            iteration += 1
            logging.debug(f"START ITERATION - {iteration}")

            # Reset the global tool call status
            tool_call_status = False

            # Invoke the graph
            updated_state = graph.invoke(initial_state, config={"configurable": {"thread_id": 1}})
            last_message = updated_state["messages"][-1]

            # Print the AI's response
            ai_response = updated_state["messages"][-1].content
            print(f"AI: {ai_response}")
            # Print the entire message
 
            with open("all_msg.txt", "a") as file:  # Use "a" to append
                file.write("======== START OF ITERATION ========\n")
                file.write(json.dumps(
                {"messages": [message.__dict__ for message in updated_state['messages']]},
                  indent=4
                 ))
                file.write("\n======== END OF ITERATION ========\n\n")
            initial_state = updated_state
            logging.debug(f"ENDING ITERATION - {iteration}")
            
            # Write the messages to the specified file
            write_all_message_types_to_txt(updated_state.values())
    except Exception as e:
        logging.error(f"Error during multi-turn conversation: {e}")
        raise
