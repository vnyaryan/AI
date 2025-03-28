from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
import sys

class ConversationState:
    state: MessagesState  # Declare the type

class InputTracker:
    """
    Custom class to track user input and exit status.
    """
    def __init__(self):
        self.exit_status = False  # Tracks if the user wants to exit

def print_all_messages(state):
    """
    Prints all messages in the state to show the conversation so far.
    """
    messages = state.get("messages", [])
    print("\n--- All Messages ---")
    if not messages:
        print("No messages in the current state.")
    else:
        for idx, msg in enumerate(messages, start=1):
            print(f"{idx}. {msg.type}: {msg.content}")
    print("--------------------\n")

def get_user_input():
    """
    Prompts the user for input and updates the exit status in the tracker if the input is "exit".

    Args:
        tracker (InputTracker): The custom input tracker to update exit status.

    Returns:
        str: The message entered by the user, or an empty string if the user wants to exit.
    """
    tracker = InputTracker()
    user_message = input("Enter your message (type 'exit' to quit): ")
    if user_message.lower() == "exit":
        tracker.exit_status = True
        print("Exiting conversation...")
        sys.exit(0)
    return user_message


def add_user_message_to_state(state, user_message):
    """
    Adds the provided user message to the MessagesState.

    Args:
        state (MessagesState): The current state to which the message will be added.
        user_message (str): The message to be added.
    """
    current_messages = state.get("messages", [])
    state["messages"] = add_messages(current_messages, [HumanMessage(content=user_message)])
    print("Message added to state!")


def initialize_and_run():
    """
    Initializes the state and tracker, then handles the user input loop.
    """
    # Create an instance of ConversationState
    conversation_state = ConversationState()

    # Assign a MessagesState instance to the state attribute
    conversation_state.state = MessagesState({"messages": []})

    try:
        while True:  # Keep running until the user decides to exit
            print("Calling get_user_input and add_user_message_to_state...")

            # Get user input using the get_user_input function
            user_message = get_user_input()

            # Add the user message to the state using add_user_message_to_state
            add_user_message_to_state(conversation_state.state, user_message)
            # Print all messages each time a new one is added
            print_all_messages(conversation_state.state)            

    except Exception as e:
        print(f"An error occurred: {e}")



def main():
    initialize_and_run()


if __name__ == "__main__":
    main()
