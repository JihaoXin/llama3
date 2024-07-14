from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Set the environment variable for PyTorch CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./HuggingFace/Meta-Llama-3-70B", local_files_only=True)

# Load model with automatic device mapping (or manually if needed)
model = AutoModelForCausalLM.from_pretrained("./HuggingFace/Meta-Llama-3-70B", local_files_only=True, device_map="auto", torch_dtype=torch.float16)

# Check device allocation of model parameters
for name, param in model.named_parameters():
    print(f"Layer: {name} - Shape: {param.shape} - Device: {param.device} - Dtype: {param.dtype}")


print(f"Start inference!")

# Example input text
input_text = "Hello, how are you?"

# Tokenize the input text and move the input tensors to the same device as the model
inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to(model.device)

# Perform inference (generate text)
outputs = model.generate(**inputs, max_new_tokens=1)

# Decode the generated tokens back into text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
