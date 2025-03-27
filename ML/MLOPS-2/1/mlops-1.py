import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from auto_gptq import AutoGPTQForCausalLM

def setup_logging():
    """
    Sets up logging for the application.

    Creates the log directory if it does not exist and configures the logging settings 
    to save logs in the specified file with detailed debug information.

    Raises:
        Exception: If there's an error while setting up logging.
    """
    try:
        log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-2\1'
        log_filename = 'mlops-1.log'

        os.makedirs(log_directory, exist_ok=True)

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(log_directory, log_filename),
            filemode='w'
        )
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        raise

# Setup logging first
setup_logging()

logging.info("Starting DeepSeek Distillation & Quantization")

# ---------------------------- Constants ----------------------------
TEACHER_MODEL = "deepseek-ai/deepseek-chat"  # Teacher model (ensure this is the correct variant)
STUDENT_MODEL = "deepseek-ai/deepseek-small"   # Student model for distillation
QUANTIZED_MODEL_PATH = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-2\1\quantized_deepseek'
DATASET_NAME = "wikitext"  # Using wikitext for causal language modeling

# ---------------------------- Load Tokenizer ----------------------------
logging.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)

# ---------------------------- Load Dataset ----------------------------
logging.info(f"Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME, "wikitext-103-raw-v1", split="train").shuffle(seed=42).select(range(5000))

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.remove_columns(["text"])  # Remove original text column for training

# ---------------------------- Load Teacher & Student Model ----------------------------
logging.info("Loading teacher model...")
teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL)

logging.info("Loading student model...")
student_model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL)

# ---------------------------- Fine-Tune the Student Model (Distillation) ----------------------------
training_args = TrainingArguments(
    output_dir=r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-2\1\distilled_deepseek',
    per_device_train_batch_size=8,
    num_train_epochs=2,
    save_steps=500,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=student_model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
)

logging.info("Fine-tuning student model via distillation...")
trainer.train()

# Save the distilled student model to the same output directory specified in training_args
student_model.save_pretrained(r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-2\1\distilled_deepseek')

# ---------------------------- Apply Quantization ----------------------------
logging.info("Applying 4-bit Quantization using GPTQ...")
quantized_model_4bit = AutoGPTQForCausalLM.from_pretrained(
    r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-2\1\distilled_deepseek',
    quantize_config={"bits": 4, "desc_act": True}  # Adjust GPTQ settings as needed
)

quantized_model_4bit.save_pretrained(QUANTIZED_MODEL_PATH)
logging.info(f"Quantized model saved at {QUANTIZED_MODEL_PATH}")

# ---------------------------- Model Evaluation ----------------------------
logging.info("Evaluating final quantized model...")
sample_text = "Deep learning models are transforming AI."
inputs = tokenizer(sample_text, return_tensors="pt")

with torch.no_grad():
    output = quantized_model_4bit.generate(**inputs, max_length=50)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
logging.info(f"Generated Text: {generated_text}")
