import os
import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set the environment variable for PyTorch CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Clear existing memory allocations
torch.cuda.empty_cache()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./HuggingFace/Meta-Llama-3-70B", local_files_only=True)

# Load model with DeepSpeed
model = AutoModelForCausalLM.from_pretrained(
    "./HuggingFace/Meta-Llama-3-70B",
    local_files_only=True,
    torch_dtype=torch.float16
)

# Read DeepSpeed configuration
ds_config = {
   "train_batch_size": 3,  # Dummy value for inference
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        }
    },
    "steps_per_print": 2000,
    "wall_clock_breakdown": False
}

# Initialize DeepSpeed for inference
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params=ds_config
)

print("Start inference!")

# Example input text
input_text = "Hello, how are you?"

# Tokenize the input text and move the input tensors to the same device as the model
inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to(model_engine.device)

# Perform inference (generate text)
with torch.no_grad():
    outputs = model_engine.generate(**inputs)

# Decode the generated tokens back into text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
