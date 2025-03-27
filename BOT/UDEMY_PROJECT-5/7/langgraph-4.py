import os
import logging
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages  
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Literal

OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Step 1: Setup Logging
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\7'
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

# Step 2: Define TypedDict Classes
class OverallState(TypedDict):
    messages: list[AnyMessage]



# Step 3: Define Node 1 - Collect One Input
def node_1(state: OverallState) -> OverallState:
    """
    Node 1: Collects a single user input and adds it to the state.
    """
    try:
        logging.debug("NODE_1  EXECUTION STARTED.")
        
        # Prompt the user for input
        user_message = input("You: ")
        
        # Check for termination condition
        if user_message.lower() == "exit":
            print("Ending conversation.")
            logging.debug("User requested to terminate the conversation.")
            os._exit(0)
            

        
        # Add the user message to the state using add_messages
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_message)]) 

         # the last added message    
        last_message = state["messages"][-1]

        # ADDING LOG DETAILS          
        logging.debug(f"User message added to state via add_messages: {state['messages']}")
        logging.debug(f"LAST ADDED MESSAGE CONTENT -  {last_message.content}")
        logging.debug(f"DETAILS OF STATE KEY IN CLASS OVERALLSTATE -  {state['messages']}")        
        logging.debug("NODE_1  EXECUTION COMPLETED.")
        return state
    except Exception as e:
        logging.error(f"Error in node_1: {e}")
        raise

# Step 4: Define Node 2 - Send Messages to LLM and Retrieve Response
def node_2(state: OverallState) -> OverallState:
    """
    Node 2: Sends the conversation history to the LLM and appends the AI's response.
    """
    try:

        logging.debug("NODE_2  EXECUTION STARTED.")
        # Initialize the OpenAI model
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

        # Ensure messages array is not empty
        if not state["messages"]:
            raise ValueError("State messages are empty. Cannot send an empty conversation to the LLM.")



        # Send the messages to the LLM
        llm_response = llm.invoke(state["messages"])

        # Add the AI's response to the state using add_messages
        state["messages"] = add_messages(state["messages"], [AIMessage(content=llm_response.content.strip())])  
        # the last added message    
        last_message = state["messages"][-1]
        
        # ADDING LOG DETAILS          
        logging.debug(f"LLM response added to state via add_messages: {state['messages']}")
        logging.debug(f"LAST ADDED MESSAGE CONTENT -  {last_message.content}")
        logging.debug(f"DETAILS OF STATE KEY IN CLASS OVERALLSTATE -  {state['messages']}")        
        logging.debug("NODE_2  EXECUTION COMPLETED.")

        
        return state
    except Exception as e:
        logging.error(f"Error in node_2: {e}")
        raise









# Step 5: Build StateGraph
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





# Step 6: Invoke Graph and Log Results
if __name__ == "__main__":
    try:
        logging.debug("Starting multi-turn conversation.")
        
        # Initialize input state
        initial_state = {"messages": []}
        
        # Initialize iteration counter
        iteration = 0
        
        while True:
            
            # Increment iteration counter
            iteration += 1

           # Log the current iteration
            logging.debug(f"START ITERATION - {iteration} ")

            # Invoke the graph with the current state
            updated_state = graph.invoke(initial_state, config={"configurable": {"thread_id": 1}})




            # Print the AI response
            ai_response = updated_state["messages"][-1].content
            print(f"AI: {ai_response}")
            
            # Update the state for the next turn
            initial_state = updated_state
           
           # Log the current iteration
            logging.debug(f"ENDING ITERATION -  {iteration} ")


        logging.debug(f"Final State: {initial_state}")
        print("Conversation ended successfully.")

    except Exception as e:
        logging.error(f"Error during multi-turn conversation: {e}")
        raise


        logging.debug(f"Final State: {initial_state}")
        print("Conversation ended successfully.")
        
        # Access and log checkpoint history
        logging.debug("Accessing checkpoint history...")
        checkpoint_history_generator = graph.get_state_history(config)
        for idx, checkpoint in enumerate(checkpoint_history_generator):
            logging.debug(f"Checkpoint {idx + 1}: {checkpoint}")
        
        # Access the final checkpoint state
        final_checkpoint_state = graph.get_state(config)
        logging.debug(f"Final checkpoint state: {final_checkpoint_state}")

    except Exception as e:
        logging.error(f"Error during multi-turn conversation: {e}")
        raise
