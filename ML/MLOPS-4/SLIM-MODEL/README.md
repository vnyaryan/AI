---
license: cc-by-sa-4.0
---

# SLIM-SUMMARY-TOOL

<!-- Provide a quick summary of what the model is/does. -->


**slim-summary-tool** is a 4_K_M quantized GGUF version of slim-summary, providing a small, fast inference implementation, to provide high-quality summarizations of complex business documents, on a small, specialized locally-deployable model with summary output structured as a python list of key points.  

The size of the self-contained GGUF model binary is 1.71 GB, which is small enough to run locally on a CPU with reasonable inference speed, and has been designed to balance high-quality with the ability to deploy on a local machine.  

The model takes as input a text passage, an optional parameter with a focusing phrase or query, and an experimental optional (N) parameter, which is used to guide the model to a specific number of items return in a summary list.  

Please see the usage notes at:  [**slim-summary**](https://huggingface.co/llmware/slim-summary) 


To pull the model via API:  

    from huggingface_hub import snapshot_download           
    snapshot_download("llmware/slim-summary-tool", local_dir="/path/on/your/machine/", local_dir_use_symlinks=False)  
    

Load in your favorite GGUF inference engine, or try with llmware as follows:

    from llmware.models import ModelCatalog  
    
    # to load the model and make a basic inference
    model = ModelCatalog().load_model("slim-summary-tool")
    response = model.function_call(text_sample)  

    # this one line will download the model and run a series of tests
    ModelCatalog().tool_test_run("slim-summary-tool", verbose=True)  


Note: please review [**config.json**](https://huggingface.co/llmware/slim-summary-tool/blob/main/config.json) in the repository for prompt wrapping information, details on the model, and full test set.  


## Model Card Contact

Darren Oberst & llmware team  

[Any questions? Join us on Discord](https://discord.gg/MhZn5Nc39h)