# step3_load_model.py (No Quantization)
import sys
sys.path.append('/Users/aaddharbhaduri/miniforge3/envs/env_2/lib/python3.12/site-packages')
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import torch

max_seq_length = 64
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Hugging Face model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token if not present

# Load model without quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float32,  # Use full precision (requires more memory)
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Save for later steps
model.save_pretrained("./temp_model")
tokenizer.save_pretrained("./temp_model")

print("Hugging Face model and tokenizer loaded without quantization and saved to temp_model.")