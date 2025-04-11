import sys
sys.path.append('/Users/aaddharbhaduri/miniforge3/envs/env_2/lib/python3.12/site-packages')
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load fine-tuned model (no device_map specified for CPU usage)
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
tokenizer.pad_token = tokenizer.eos_token

# Ensure model is on CPU
model.to("cpu")

# Save final model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Test with a sample input
input_text = "What is abbreviation of DoF?"
inputs = tokenizer(input_text, return_tensors="pt").to("cpu")  # Ensure input is on CPU
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
print("Test Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

print("Model saved and tested successfully.")
