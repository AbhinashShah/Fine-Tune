import sys
sys.path.append('/Users/aaddharbhaduri/miniforge3/envs/env_2/lib/python3.12/site-packages')
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch
from prepare_data import pdf_texts
import json

# # Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./temp_model", device_map=None)  # Disable device_map
tokenizer = AutoTokenizer.from_pretrained("./temp_model")
tokenizer.pad_token = tokenizer.eos_token

# Function to prepare data from both sources
def prepare_combined_dataset(json_path, pdf_texts_list):
    # Load JSON dataset
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    json_texts = [item["question"] + " " + item["answer"] for item in json_data["text"]]
    
    # Create dataset from combined text
    combined_texts = json_texts + pdf_texts_list
    combined_dataset = Dataset.from_dict({"text": combined_texts})
    
    return combined_dataset

# Load and combine datasets
dataset = prepare_combined_dataset("dataset.json", pdf_texts)  # pdf_texts should be passed as a list

# Tokenize dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Load LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Training arguments for CPU
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=20,
    warmup_steps=20,
    logging_steps=10,
    save_steps=10,
    optim="adamw_torch",  # Use standard AdamW optimizer (no 8bit)
    fp16=False,  # Ensure no mixed precision
    report_to="none",
    no_cuda=True,  # Disable CUDA (forces CPU usage)
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=lambda data: {
        "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in data]),
        "attention_mask": torch.stack([torch.tensor(x["attention_mask"]) for x in data]),
        "labels": torch.stack([torch.tensor(x["input_ids"]) for x in data]),  # Causal LM uses input as labels
    },
)

# Train
trainer.train()

print("Training completed. Proceed to step 6.")
