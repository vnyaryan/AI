from llmware.models import ModelCatalog

def discover_slim_models():
    tools = ModelCatalog().list_llm_tools()
    tool_map = ModelCatalog().get_llm_fx_mapping()

    print("\nList of SLIM model tools (GGUF) in the ModelCatalog\n")
    for i, tool in enumerate(tools):
        model_card = ModelCatalog().lookup_model_card(tool_map[tool])
        print(f"{i} - tool: {tool} - model_name: {model_card['model_name']}")

discover_slim_models()
