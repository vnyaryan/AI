import os
import logging
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompt_values import ChatPromptValue
from langgraph.checkpoint.memory import MemorySaver 

OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# Step 1: Setup Logging
def setup_logging():
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-5\5'
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

class OverallState(TypedDict):
    system_message_template: str
    human_message_template: str
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
def node_1(state: dict) -> InputState:
    """
    Node 1: Collects input from the user for system_message_template and human_message_template
    and validates the collected data.
    """
    try:
        logging.debug("Executing node_1 and asking for user input.")

        # Extract system_message_template and human_message_template from state
        system_message_template = state["system_message_template"]
        human_message_template = state["human_message_template"]


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
        # Log or print the entire response for inspection
        logging.debug(f"Full LLM response: {llm_response}")

        # Validate the result
        validate_llm_response_state(result)

        logging.debug(f"node_6 output: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in node_6: {e}")
        raise

# Build StateGraph
builder = StateGraph(
    OverallState,
    input=InputState,
    output=LLMResponseState
)

# Add Nodes
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_node("node_4", node_4)
builder.add_node("node_5", node_5)
builder.add_node("node_6", node_6)

# Add Edges
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", "node_4")
builder.add_edge("node_4", "node_5")
builder.add_edge("node_5", "node_6")
builder.add_edge("node_6", END)

# Initialize Memory Saver
prompt_memory_saver = MemorySaver()

# Compile Graph with Checkpointer
graph = builder.compile(checkpointer=prompt_memory_saver)

# Invoke Graph and Log Results
if __name__ == "__main__":
    try:
        logging.debug("Starting graph invocation.")
        initial_input = {
            "system_message_template": "Act as a langgraph expert.",
            "human_message_template": "Explain LangChain."
        }
        config = {"configurable": {"thread_id": "1"}}  # Checkpointer config
        result = graph.invoke(initial_input, config)
        logging.debug(f"Graph result: {result}")
        print("Final LLM Response:", result["llm_response"])

        # Access and log checkpoints
        logging.debug("Accessing checkpoint history...")
        checkpoint_history_generator = graph.get_state_history(config)
        for idx, checkpoint in enumerate(checkpoint_history_generator):
            logging.debug(f"Checkpoint {idx + 1}: {checkpoint}")
            

        # Access final checkpoint state
        final_state = graph.get_state(config)
        logging.debug(f"Final checkpoint: {final_state}")
        

    except Exception as e:
        logging.error(f"Error during graph invocation: {e}")