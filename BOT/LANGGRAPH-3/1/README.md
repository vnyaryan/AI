---
license: apache-2.0  
inference: false  
---

# bling-phi-3-gguf

<!-- Provide a quick summary of what the model is/does. -->

bling-phi-3-gguf is part of the BLING ("Best Little Instruct No-GPU") model series, RAG-instruct trained for fact-based question-answering use cases on top of a Microsoft Phi-3 base model.


### Benchmark Tests  

Evaluated against the benchmark test:   [RAG-Instruct-Benchmark-Tester](https://www.huggingface.co/datasets/llmware/rag_instruct_benchmark_tester)  
1 Test Run (with temperature = 0.0 and sample = False) with 1 point for correct answer, 0.5 point for partial correct or blank / NF, 0.0 points for incorrect, and -1 points for hallucinations.  

--**Accuracy Score**:  **100.0** correct out of 100  
--Not Found Classification:  95.0%  
--Boolean:  97.5%  
--Math/Logic:  80.0%  
--Complex Questions (1-5):  4 (Above Average - multiple-choice, causal)  
--Summarization Quality (1-5):  4 (Above Average)  
--Hallucinations:  No hallucinations observed in test runs.  

For test run results (and good indicator of target use cases), please see the files ("core_rag_test" and "answer_sheet" in this repo).  

Note: compare results with [bling-phi-2](https://www.huggingface.co/llmware/bling-phi-2-v0), and [dragon-mistral-7b](https://www.huggingface.co/llmware/dragon-mistral-7b-v0).  


### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** llmware
- **Model type:** bling-rag-instruct 
- **Language(s) (NLP):** English
- **License:** Apache 2.0
- **Finetuned from model:** Microsoft Phi-3

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

The intended use of BLING models is two-fold:

1.  Provide high-quality RAG-Instruct models designed for fact-based, no "hallucination" question-answering in connection with an enterprise RAG workflow.

2.  BLING models are fine-tuned on top of leading base foundation models, generally in the 1-3B+ range, and purposefully rolled-out across multiple base models to provide choices and "drop-in" replacements for RAG specific use cases.


### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

BLING is designed for enterprise automation use cases, especially in knowledge-intensive industries, such as financial services,
legal and regulatory industries with complex information sources.  

BLING models have been trained for common RAG scenarios, specifically:   question-answering, key-value extraction, and basic summarization as the core instruction types
without the need for a lot of complex instruction verbiage - provide a text passage context, ask questions, and get clear fact-based responses.


## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

BLING models are designed to operate with grounded sources, e.g., inclusion of a context passage in the prompt, and will not yield consistent or positive results if open-context prompting in which you are looking for the model to draw upon potential background knowledge of the world - in fact, it is likely that the BLING will respond with a simple "Not Found." to an open context query.  

Any model can provide inaccurate or incomplete information, and should be used in conjunction with appropriate safeguards and fact-checking mechanisms.


## How to Get Started with the Model

To pull the model via API:  

    from huggingface_hub import snapshot_download           
    snapshot_download("llmware/bling-phi-3-gguf", local_dir="/path/on/your/machine/", local_dir_use_symlinks=False)  
    
Load in your favorite GGUF inference engine, or try with llmware as follows:

    from llmware.models import ModelCatalog  
    
    # to load the model and make a basic inference
    model = ModelCatalog().load_model("llmware/bling-phi-3-gguf", temperature=0.0, sample=False)
    response = model.inference(query, add_context=text_sample)  

Details on the prompt wrapper and other configurations are on the config.json file in the files repository.  

## Model Card Contact

Darren Oberst & llmware team
