"""
Script Logic:
-------------
This script is a modular pipeline that takes user inputs, processes them into structured prompts, 
sends the prompts to OpenAI's ChatGPT model using LangChain, and retrieves a response. The process
is divided into six distinct nodes, each performing a specific function. Logging and validation 
ensure that the data flows correctly through the nodes.

Steps:
1. **Node 1**: Collects user input for `system_message_template` and `human_message_template`.
2. **Node 2**: Converts the input templates into structured prompt templates using LangChain.
3. **Node 3**: Combines the structured prompt templates into a unified `ChatPromptTemplate`.
4. **Node 4**: Formats the combined template into a complete prompt by replacing placeholders.
5. **Node 5**: Converts the formatted prompt into a list of messages for the LLM.
6. **Node 6**: Sends the list of messages to OpenAI's ChatGPT and retrieves the response.

Input:
------
- **system_message_template**: A system message that guides the behavior of the assistant.
- **human_message_template**: A user query or task for the assistant.

Output:
-------
- The final response from OpenAI's ChatGPT, which answers the user's query or performs the task.

Example:
--------
Input:
  system_message_template = "Act as a language processing expert."
  human_message_template = "Explain LangChain."

Output:
  LLM Response: "LangChain is a framework for building applications powered by large language models (LLMs)."
"""

import os
import logging
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompt_values import ChatPromptValue

OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Step 1: Setup Logging
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\5'
        log_filename = 'langgraph-1.log'

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
class InputState(TypedDict):
    system_message_template: str
    human_message_template: str

class PromptTemplates(TypedDict):
    system_message_prompt_template: SystemMessagePromptTemplate
    human_message_prompt_template: HumanMessagePromptTemplate

class UnifiedPromptState(TypedDict):
    chat_prompt_template: ChatPromptTemplate

class FormattedPromptState(TypedDict):
    formatted_prompt: ChatPromptValue

class FinalPromptMessagesState(TypedDict):
    final_prompt_messages: list


class LLMResponseState(TypedDict):
    llm_response: str

# Step 3: General Validation Function
def validate_typed_dict(data: dict, expected_type: TypedDict):
    """
    Validates if a dictionary matches the structure and types defined in a TypedDict.
    """
    for key, value_type in expected_type.__annotations__.items():
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in data: {data}")
        if not isinstance(data[key], value_type):
            raise TypeError(f"Key '{key}' has incorrect type: expected {value_type}, got {type(data[key])}")
    logging.info(f"Validation successful for data: {data}")

# Step 4: Specific Validation Functions
def validate_input_state(data: dict):
    validate_typed_dict(data, InputState)

def validate_prompt_templates(data: dict):
    validate_typed_dict(data, PromptTemplates)

def validate_unified_prompt_state(data: dict):
    validate_typed_dict(data, UnifiedPromptState)

def validate_formatted_prompt_state(data: dict):
    validate_typed_dict(data, FormattedPromptState)

def validate_final_prompt_messages_state(data: dict):
    validate_typed_dict(data, FinalPromptMessagesState)

def validate_llm_response_state(data: dict):
    validate_typed_dict(data, LLMResponseState)

# Step 5: Define Node 1 - Collect Input
def node_1() -> InputState:
    """
    Node 1: Collects input from the user for system_message_template and human_message_template
    and validates the collected data.
    """
    try:
        logging.debug("Executing node_1 and asking for user input.")

        # Ask user for input
        system_message_template = input("Enter the system message template: ")
        human_message_template = input("Enter the human message template: ")

        # Construct the InputState dictionary
        input_state = {
            "system_message_template": system_message_template,
            "human_message_template": human_message_template,
        }

        # Validate the input state
        validate_input_state(input_state)

        logging.debug(f"node_1 output: {input_state}")
        return input_state
    except Exception as e:
        logging.error(f"Error in node_1: {e}")
        raise

# Step 6: Define Node 2 - Convert Templates to Prompt Templates
def node_2(state: InputState) -> PromptTemplates:
    """
    Node 2: Converts the user-provided templates into structured prompt templates.
    """
    try:
        logging.debug(f"Executing node_2 with state: {state}")

        # Convert templates into prompt templates
        system_message_prompt_template = SystemMessagePromptTemplate.from_template(state["system_message_template"])
        human_message_prompt_template = HumanMessagePromptTemplate.from_template(state["human_message_template"])

        # Construct the PromptTemplates dictionary
        result = {
            "system_message_prompt_template": system_message_prompt_template,
            "human_message_prompt_template": human_message_prompt_template,
        }

        # Validate the result
        validate_prompt_templates(result)

        logging.debug(f"node_2 output: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in node_2: {e}")
        raise

# Step 7: Define Node 3 - Combine Prompt Templates into ChatPromptTemplate
def node_3(state: PromptTemplates) -> UnifiedPromptState:
    """
    Node 3: Combines system and human structured prompt templates into a unified ChatPromptTemplate.
    """
    try:
        logging.debug(f"Executing node_3 with state: {state}")

        # Combine templates into a ChatPromptTemplate
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [state["system_message_prompt_template"], state["human_message_prompt_template"]]
        )

        # Construct the UnifiedPromptState dictionary
        result = {
            "chat_prompt_template": chat_prompt_template
        }

        # Validate the result
        validate_unified_prompt_state(result)

        logging.debug(f"node_3 output: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in node_3: {e}")
        raise

# Step 8: Define Node 4 - Format Prompt
def node_4(state: UnifiedPromptState) -> FormattedPromptState:
    """
    Node 4: Formats the prompt by replacing placeholders with specific query/task from the user.
    """
    try:
        logging.debug(f"Executing node_4 with state: {state}")

        # Format the chat prompt template
        formatted_prompt = state["chat_prompt_template"].format_prompt()

        # Construct the FormattedPromptState dictionary
        result = {
            "formatted_prompt": formatted_prompt
        }

        # Validate the result
        validate_formatted_prompt_state(result)

        logging.debug(f"node_4 output: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in node_4: {e}")
        raise

# Step 9: Define Node 5 - Convert Formatted Prompt to Messages
def node_5(state: FormattedPromptState) -> FinalPromptMessagesState:
    """
    Node 5: Converts the formatted prompt into a list of messages.
    """
    try:
        logging.debug(f"Executing node_5 with state: {state}")

        # Convert the formatted prompt into messages
        final_prompt_messages = state["formatted_prompt"].to_messages()

        # Construct the FinalPromptMessagesState dictionary
        result = {
            "final_prompt_messages": final_prompt_messages
        }

        # Validate the result
        validate_final_prompt_messages_state(result)

        logging.debug(f"node_5 output: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in node_5: {e}")
        raise

def node_6(state: FinalPromptMessagesState) -> LLMResponseState:
    """
    Node 6: Initializes the OpenAI model, sends the prompt messages, and returns the response.
    """
    try:
        logging.debug(f"Executing node_6 with state: {state}")

        # Initialize the OpenAI model
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

        # Send the prompt messages to the LLM
        llm_response = llm.invoke(state["final_prompt_messages"])

        # Construct the LLMResponseState dictionary
        result = {
            "llm_response": llm_response.content.strip()  # Extract response content
        }

        # Validate the result
        validate_llm_response_state(result)

        logging.debug(f"node_6 output: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in node_6: {e}")
        raise

# Step 10: Main Execution
if __name__ == "__main__":
    try:
        logging.debug("Starting script execution.")

        # Execute Node 1
        input_state = node_1()

        # Execute Node 2
        prompt_templates = node_2(input_state)

        # Execute Node 3
        unified_prompt_state = node_3(prompt_templates)

        # Execute Node 4
        formatted_prompt_state = node_4(unified_prompt_state)

        # Execute Node 5
        final_prompt_messages_state = node_5(formatted_prompt_state)

        # Execute Node 6
        llm_response_state = node_6(final_prompt_messages_state)    

        print("LLM Response:", llm_response_state["llm_response"])
        logging.debug(f"Final Output: {llm_response_state}")

    except Exception as e:
        logging.error(f"Error in script execution: {e}")
