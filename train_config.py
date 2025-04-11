# step4_train_config.py
import sys
sys.path.append('/Users/aaddharbhaduri/miniforge3/envs/env_2/lib/python3.12/site-packages')
from peft import LoraConfig
from transformers import TrainingArguments

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=20,
    warmup_steps=20,
    logging_steps=10,
    save_steps=10,
    optim="adamw_8bit",
    fp16=True,
    report_to="none",
)

# Save configurations
with open("training_config.txt", "w") as f:
    f.write(str(training_args))

print("Training configuration completed. Proceed to step 5.")