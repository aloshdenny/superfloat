import modal
import torch
import time
import psutil
import gc
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set your Hugging Face token
hf_token = "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"
os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

# Define the mapping for bit-width
def map_bitwidth(bits):
    if 4 <= bits <= 7:
        return 4
    elif 8 <= bits <= 15:
        return 8
    else:
        return 16

# Mapping bit-width to model names
model_mapping = {
    "Qwen/Qwen2.5-0.5B": {
        4: "Qwen/Qwen2.5-0.5B",
        8: "Qwen/Qwen2.5-0.5B",
        16: "Qwen/Qwen2.5-0.5B"
    },
    "Qwen/Qwen2.5-1.5B": {
        4: "Qwen/Qwen2.5-1.5B",
        8: "Qwen/Qwen2.5-1.5B",
        16: "Qwen/Qwen2.5-1.5B"
    },
    "meta-llama/Llama-3.2-1B": {
        4: "meta-llama/Llama-3.2-1B",
        8: "meta-llama/Llama-3.2-1B",
        16: "meta-llama/Llama-3.2-1B"
    }
}

# Function to quantize the model
def absmax_quantize(tensor, bitwidth):
    scale = torch.max(torch.abs(tensor))
    q_tensor = torch.round(tensor / scale * (2**(bitwidth - 1) - 1))
    deq_tensor = q_tensor / (2**(bitwidth - 1) - 1) * scale
    return deq_tensor

def zero_mean_quantize(tensor, bitwidth):
    scale = torch.max(torch.abs(tensor - tensor.mean()))
    q_tensor = torch.round((tensor - tensor.mean()) / scale * (2**(bitwidth - 1) - 1))
    deq_tensor = q_tensor / (2**(bitwidth - 1) - 1) * scale + tensor.mean()
    return deq_tensor

def load_model(model_name, bitwidth, quantization_type, device):
    # Load the model in half-precision (torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    
    # Apply quantization
    for param in model.parameters():
        if quantization_type == 'absmax':
            param.data = absmax_quantize(param.data, bitwidth).to(torch.float16)  # Ensure quantization output is float16
        elif quantization_type == 'zero_mean':
            param.data = zero_mean_quantize(param.data, bitwidth).to(torch.float16)  # Ensure quantization output is float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    return model, tokenizer

def measure_performance(model, tokenizer, input_text, device):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors='pt').to(device)
    
    # Ensure input_ids remains as torch.long (integer)
    inputs = {k: v.to(torch.long) if k == "input_ids" else v.to(torch.float16) for k, v in inputs.items()}
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    end_time = time.time()
    inference_time = end_time - start_time
    memory_usage = psutil.Process().memory_info().rss / (1024 ** 2)  # in MB
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return inference_time, memory_usage, generated_text

def calculate_perplexity(model, tokenizer, input_text, device):
    inputs = tokenizer(input_text, return_tensors='pt').to(device)
    max_length = inputs.input_ids.size(1)
    with torch.no_grad():
        outputs = model(**inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs.input_ids[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = torch.exp(loss).item()
    return perplexity

# Define request models
class ModelRequest(BaseModel):
    model_name: str
    quantization_bits: int
    quantization_type: str
    input_text: str

# Create a Modal Dict for persistent storage
results_dict = modal.Dict.from_name("emelinlabs-results", create_if_missing=True)

# Create a FastAPI app
app_fastapi = FastAPI()

# Modal setup
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi", "uvicorn", "transformers", "torch", "psutil", "pydantic")
)

app = modal.App(name="emelinlabs-runner", image=image)

# POST endpoint
@app.function(
    gpu="A100",  # Specify the GPU type (e.g., "A10G", "A100", "H100")
    timeout=86400,  # Timeout in seconds (1 day = 86400 seconds)
    allow_concurrent_inputs=100  # Allow concurrent requests
)
@modal.web_endpoint(method="POST")
def run_inference(request: ModelRequest):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = request.model_name
    quantization_bits = request.quantization_bits
    quantization_type = request.quantization_type
    input_text = request.input_text

    # Generate a unique ID for this request
    request_id = str(uuid.uuid4())

    # Load original model
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    
    # Load and quantize model
    effective_bits = map_bitwidth(quantization_bits)
    quantized_model_name = model_mapping[model_name][effective_bits]
    quantized_model, _ = load_model(quantized_model_name, effective_bits, quantization_type, device)

    # Measure performance for original model
    orig_inference_time, orig_memory_usage, orig_text = measure_performance(original_model, tokenizer, input_text, device)
    orig_perplexity = calculate_perplexity(original_model, tokenizer, input_text, device)

    # Measure performance for quantized model
    quant_inference_time, _, quant_text = measure_performance(quantized_model, tokenizer, input_text, device)
    quant_perplexity = calculate_perplexity(quantized_model, tokenizer, input_text, device)

    # Calculate memory usage for quantized model
    quant_memory_usage = (effective_bits / 16.0) * orig_memory_usage

    # Calculate differences
    speed_diff = (orig_inference_time - quant_inference_time) / orig_inference_time * 100
    memory_savings = (orig_memory_usage - (effective_bits / 16.0) * quant_memory_usage) / orig_memory_usage * 100
    perplexity_diff = quant_perplexity - orig_perplexity

    # Store results in Modal Dict
    results_dict[request_id] = {
        "original": {
            "text": orig_text,
        },
        "quantized": {
            "text": quant_text,
        },
        "comparison": {
            "speed_diff": speed_diff,
            "memory_savings": memory_savings,
            "perplexity_diff": perplexity_diff
        }
    }

    # Clean up to free memory
    del original_model
    del quantized_model
    gc.collect()
    torch.cuda.empty_cache()

    return {"request_id": request_id}

# GET endpoint
@app.function()
@modal.web_endpoint()
def get_result(request_id: str):
    result = results_dict.get(request_id, None)
    if result:
        return result
    else:
        return {"error": "Request ID not found"}

# Health check endpoint
@app.function()
@modal.web_endpoint()
def health_check():
    return {"status": "active"}