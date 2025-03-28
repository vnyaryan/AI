import os
import logging
from langgraph.graph.message import add_messages
from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from typing_extensions import TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage



class ConversationState:
    """
    Stores conversation state.
    For demonstration, 'state' is simply a dict with a 'messages' key.
    'exit_status' indicates if the user typed 'exit'.
    """
    state: dict
    exit_status: bool = False


def get_user_input(conversation_state: ConversationState) -> str:
    user_input = input("Enter your message (or type 'exit' to quit): ")
    if user_input.strip().lower() == "exit":
        conversation_state.exit_status = True
    return user_input


def add_message_to_state(conversation_state: ConversationState, user_message: str):
    """
    Uses add_messages from langgraph.graph.message to append a new HumanMessage.
    'conversation_state.state["messages"]' is treated like a list of messages.
    """
    # 1) Get the current list of messages (a list).
    current_messages = conversation_state.state["messages"]

    # 2) Use add_messages(...) to append the new message.
    #    add_messages returns the updated list, so we store it back into state["messages"].
    conversation_state.state["messages"] = add_messages(
        current_messages,
        [HumanMessage(content=user_message)]
    )

    # 3) Print all messages so far, assuming each message object has .content
    print("\nCurrent messages in conversation state:")
    for idx, msg in enumerate(conversation_state.state["messages"], start=1):
        print(f"{idx}. {msg.content}")
    print()


def conversation_loop(conversation_state: ConversationState):
    """
    Keeps asking for user input and calling add_message_to_state,
    until user types 'exit'.
    """
    while True:
        user_message = get_user_input(conversation_state)
        if conversation_state.exit_status:
            print("Exiting the conversation loop (but not the entire script).")
            break

        add_message_to_state(conversation_state, user_message)

    print("Conversation loop has ended. The script can continue with other logic here.")


def print_conversation_summary(conversation_state: ConversationState):
    """
    Prints a final summary of messages and the exit_status.
    """
    print("\n--- Conversation Ended ---")
    print("Final list of messages:")
    for idx, msg in enumerate(conversation_state.state["messages"], start=1):
        print(f"{idx}. {msg.content}")
    print(f"Exit status: {conversation_state.exit_status}")


def run_conversation_and_print():
    """
    Initializes conversation_state.state as a dict containing an empty list for "messages".
    Then starts the conversation loop and prints the summary.
    """
    conversation_state = ConversationState()

    # Instead of a specialized MessagesState object, we just store a dict with a key "messages".
    conversation_state.state = {"messages": []}

    conversation_loop(conversation_state)
    print_conversation_summary(conversation_state)


if __name__ == "__main__":
    run_conversation_and_print()
