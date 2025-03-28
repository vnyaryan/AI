import os
import logging
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages  
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Step 1: Setup Logging
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\6'
        log_filename = 'langgraph-4.log'

        os.makedirs(log_directory, exist_ok=True)

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='w')
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")

setup_logging()

# Step 2: Define TypedDict Classes
class OverallState(TypedDict):
    messages: list[AnyMessage]

# Step 3: Define Tools
@tool
def add_two_numbers(num1: float, num2: float) -> float:
    """Adds two numbers and returns the result."""
    return num1 + num2

# Step 4: Create ReAct Agent
tools = [add_two_numbers]
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
react_agent = create_react_agent(model, tools=tools)

# Step 5: Define Node 1 - Collect One Input
def node_1(state: OverallState) -> OverallState:
    """
    Node 1: Collects a single user input and adds it to the state.
    """
    try:
        logging.debug("Executing node_1 and collecting user input.")
        
        # Prompt the user for input
        user_message = input("You: ")
        
        # Check for termination condition
        if user_message.lower() == "exit":
            print("Ending conversation.")
            return None
        
        # Add the user message to the state using add_messages
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_message)])  
        logging.debug(f"User message added to state via add_messages: {state['messages']}")
        
        return state
    except Exception as e:
        logging.error(f"Error in node_1: {e}")
        raise

# Step 6: Define Node 2 - Send Messages to ReAct Agent
def node_2(state: OverallState) -> OverallState:
    """
    Node 2: Sends the conversation history to the ReAct agent,
    which decides whether to call a tool or generate a response.
    Avoids echoing user input by ensuring the AI response is meaningful.
    """
    try:
        logging.debug(f"Executing node_2 with state: {state}")

        # Ensure messages array is not empty
        if not state["messages"]:
            raise ValueError("State messages are empty. Cannot send an empty conversation to the agent.")

        # Send messages to the ReAct agent
        inputs = {"messages": state["messages"]}
        for response in react_agent.stream(inputs, stream_mode="values"):
            message = response["messages"][-1]

            # Avoid echoing user input
            user_message = state["messages"][-1].content.strip()
            if isinstance(message, tuple):
                logging.debug(f"Tool response: {message}")
                tool_response = f"Tool Response: {message}"
                state["messages"] = add_messages(state["messages"], [AIMessage(content=tool_response)])
            else:
                ai_response = message.content.strip()
                if ai_response == user_message:
                    # Modify AI response to make it meaningful
                    ai_response = "I noticed you repeated your input. How can I assist you further?"

                # Add AI's response to the state
                logging.debug(f"AI response added: {ai_response}")
                state["messages"] = add_messages(state["messages"], [AIMessage(content=ai_response)])
        
        return state
    except Exception as e:
        logging.error(f"Error in node_2: {e}")
        raise

# Step 7: Build StateGraph
builder = StateGraph(
    OverallState,
    input=OverallState,
    output=OverallState
)

# Add Nodes
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)

# Add Edges
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

# Initialize Memory Saver
prompt_memory_saver = MemorySaver()

# Compile Graph with MemorySaver
graph = builder.compile(checkpointer=prompt_memory_saver)

# Step 8: Invoke Graph and Log Results
if __name__ == "__main__":
    try:
        logging.debug("Starting multi-turn conversation.")
        
        # Initialize input state
        initial_state = {"messages": []}

        while True:
            # Collect user input
            updated_state = node_1(initial_state)

            # Check if conversation ended
            if updated_state is None:
                break

            # Send messages to ReAct agent and get a response
            final_state = node_2(updated_state)

            # Print the AI response
            ai_response = final_state["messages"][-1].content
            print(f"AI: {ai_response}")
            
            # Update the state for the next turn
            initial_state = final_state

        logging.debug(f"Final State: {initial_state}")
        print("Conversation ended successfully.")
    except Exception as e:
        logging.error(f"Error during multi-turn conversation: {e}")
        raise
