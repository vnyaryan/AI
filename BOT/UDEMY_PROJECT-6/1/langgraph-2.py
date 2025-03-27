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

tool_call_status = False
# Logging Setup
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-6\langgraph\1'
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



# Define the directory path for the JSON file
JSON_FILE_DIRECTORY = r"C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-6\langgraph\1"
JSON_FILE_NAME = "structured_response.json"
JSON_FILE_PATH = os.path.join(JSON_FILE_DIRECTORY, JSON_FILE_NAME)

# Function to Generate and Append Structured Response
def generate_structured_response(state, output_file=JSON_FILE_PATH):
    """
    Generates a structured response based on the state, invokes LLM, and appends the output to a file.

    Args:
        state (MessagesState): The current state of messages and context.
        output_file (str): Path to the file where the structured response will be written.

    Side Effects:
        Appends the structured response to the specified file.
    """
    try:
        # Ensure the directory exists
        os.makedirs(JSON_FILE_DIRECTORY, exist_ok=True)

        # Define schema for structured response
        structured_response_schema = {
            "status": "success",
            "tool_message": {
                "content": None,
                "tool_name": None,
            },
            "llm_response": {
                "content": None,
            },
        }

        # Get the latest ToolMessage
        latest_tool_message = get_last_tool_message(state["messages"])
        if latest_tool_message:
            structured_response_schema["tool_message"]["content"] = latest_tool_message.content
            structured_response_schema["tool_message"]["tool_name"] = getattr(latest_tool_message, "name", None)
            logging.debug(f"Latest ToolMessage Content: {latest_tool_message.content}")

        # Invoke the LLM with the current state
        response = llm.invoke(state["messages"])
        structured_response_schema["llm_response"]["content"] = response.content
        logging.info("LLM invoked successfully for structured response.")

        # Check if the file exists and has existing JSON content
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            try:
                with open(output_file, "r") as file:
                    existing_data = json.load(file)
                    logging.debug(f"Existing data loaded from {output_file}.")
            except json.JSONDecodeError:
                logging.warning(f"File {output_file} contains invalid JSON. Initializing with an empty list.")
                existing_data = []
        else:
            existing_data = []

        # Ensure the existing data is a list
        if not isinstance(existing_data, list):
            logging.warning(f"Unexpected JSON structure in {output_file}. Overwriting with a new list.")
            existing_data = []

        # Append the new structured response
        existing_data.append(structured_response_schema)

        # Write the updated data back to the file
        with open(output_file, "w") as file:
            json.dump(existing_data, file, indent=4)
        logging.debug(f"Structured response appended and written to file: {output_file}")

    except Exception as e:
        logging.error(f"Error generating structured response: {e}")
        # Handle errors, e.g., log them or raise exceptions



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

#LAST TOOL MESSAGE -
def get_last_tool_message(messages):
    """
    Retrieves the last ToolMessage from a list of messages.
    
    Args:
        messages (list): List of message objects (e.g., ToolMessage, HumanMessage, etc.).

    Returns:
        ToolMessage: The last ToolMessage found in the list, or None if no ToolMessage exists.
    """
    # Initialize variable to store the last ToolMessage
    last_tool_message = None

    # Iterate through all messages in order
    for message in messages:
        if isinstance(message, ToolMessage):
            last_tool_message = message  # Update with the current ToolMessage

    # Return the last ToolMessage or None if not found
    return last_tool_message


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
    logging.debug("HITL EXECUTION STARTED.")
    global tool_call_status
    if not tool_calls:
        logging.debug("NO Tool Calls Detected")
        logging.debug("HITL EXECUTION COMPLETD.")
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
        tool_call_status = True
        logging.debug("Tool execution approved -HITL EXECUTION COMPLETD.")
        return "tools"  # Proceed to tool execution
    elif user_choice == "2":
        print("Tool execution skipped.")
        logging.debug("Tool execution skipped - HITL EXECUTION COMPLETD.")
        return "node3"  # Skip execution
    else:
        print("Invalid choice. Skipping tool execution.")
        return "node3"
        logging.debug("Invalid choice. Skipping tool execution - HITL EXECUTION COMPLETD.")



# Node 3: Process After Tool Execution
def node3(state: MessagesState):
    """
    Node 3: Processes tool execution results and invokes the LLM only if needed.
    """
    try:
        logging.debug("NODE3 EXECUTION STARTED.")
        global tool_call_status

        # Check if any tool calls exist in the messages
        #tool_calls_detected = any(isinstance(message, ToolMessage) for message in state["messages"])
        
        # if tool_calls_detected:
        #     # Tool calls detected, proceed with LLM invocation
        #     logging.debug("Tool call detected. Preparing to invoke LLM.")
        if tool_call_status:
            logging.debug("Tool call status is True. Invoking LLM.")

            # Get the latest ToolMessage
            latest_tool_message = get_last_tool_message(state["messages"])
            if latest_tool_message:
                logging.debug(f"Latest ToolMessage before invoking LLM: {latest_tool_message.content}")
                generate_structured_response(state)
            
            # Invoke LLM
            response = llm.invoke(state["messages"])
            logging.info("LLM invoked successfully.")
            logging.debug(f"LLM response: {response}")
            logging.debug("NODE3 EXECUTION COMPLETED.")
            return {"messages": [response]}
        else:
            # No tool calls detected, skip LLM invocation
            logging.debug("No tool calls detected. Skipping LLM invocation.")
            logging.debug("NODE3 EXECUTION COMPLETED.")
            return {"messages": state["messages"]}  # Return unchanged state

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
def write_all_message_types_to_txt(state_values, output_file="C:\\Users\\ARYAVIN\\Documents\\GitHub\\BOT\\UDEMY_PROJECT-6\\langgraph\\1\\msg_details.txt"):
    """
    Processes and writes the content of all message types (HumanMessage, AIMessage, ToolMessage, etc.)
    to a text file with separators
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

Your task is to assist the user by either:
1. Using the `get_temperature` or `get_population` tool when the user query involves getting temperature or population details.
2. Responding directly using your general knowledge for any other queries or topics not related to temperature or population.

**Guidelines for Tool Usage:**
- If the user query includes keywords like "weather," "temperature," or anything related to the climate of a location, use the `get_temperature` tool.
- If the user query includes keywords like "population," "people," "inhabitants," or anything related to the number of people in a location or country, use the `get_population` tool.
- For all other topics, rely on your own knowledge without invoking any tool.

| **Keyword**      | **Function**       | **Description**                                  |
|-------------------|--------------------|-------------------------------------------------|
| "weather"         | get_temperature    | Provide temperature details for a location.     |
| "temperature"     | get_temperature    | Provide temperature details for a location.     |
| "population"      | get_population     | Provide population of a location or country.    |
| "people"          | get_population     | Provide population of a location or country.    |
| "inhabitants"     | get_population     | Provide population of a location or country.    |

**Important Notes:**
- Always determine the context of the conversation.
- Use tools only when the user explicitly or implicitly requests information about temperature or population.
- For all other queries, provide helpful responses using your general knowledge and reasoning abilities.

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
            # Reset tool_call_status after each iteration
            #global tool_call_status
            tool_call_status = False
            logging.debug("Tool call status reset to False after iteration.")
            
            # Write the messages to the specified file
            write_all_message_types_to_txt(updated_state.values())
            logging.debug(f"ENDING ITERATION - {iteration}")
            
    except Exception as e:
        logging.error(f"Error during multi-turn conversation: {e}")
        raise
