import modal
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gc
from concurrent.futures import ThreadPoolExecutor

# Define the Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("flask", "transformers", "torch", "fastapi", "uvicorn", "pydantic")
)

# Create a Modal app
app = modal.App(name="streaming-llm-backend", image=image)

# Create a Modal Volume for caching model weights
cache_volume = modal.Volume.from_name("model-cache", create_if_missing=True)

# Global variables to store the tokenizer, model, KV cache, and input buffer
tokenizer = None
model = None
past_key_values = None
input_buffer = ""
previous_input_ids = torch.tensor([], dtype=torch.long).to("cuda")  # Initialize as an empty tensor

# Load the tokenizer and model inside the Modal container
def load_model():
    global tokenizer, model

    model_name = "meta-llama/Llama-3.2-3B"
    cache_dir = "/cache"

    # Download and cache the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, cache_dir=cache_dir
    ).to("cuda")

    # Perform a dummy inference to ensure everything is cached and ready
    inputs = tokenizer("Warm-up request", return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(inputs.input_ids, max_new_tokens=1)

    print("Model warm-up completed.")

    # Persist the cache to the Modal Volume
    cache_volume.commit()

def tokenize_incrementally(tokenizer, input_buffer, previous_input_ids):
    # Tokenize only the new portion of the input
    new_input = input_buffer[len(previous_input_ids):]
    new_input_ids = tokenizer.encode(new_input, return_tensors="pt", add_special_tokens=False).to("cuda")
    
    # Append new tokens to the previous input IDs
    updated_input_ids = torch.cat([previous_input_ids, new_input_ids], dim=-1)
    return updated_input_ids

def update_kv_cache(model, input_ids, past_key_values):
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
        updated_past_key_values = outputs.past_key_values
    return updated_past_key_values

def should_generate_output(input_buffer):
    # Trigger generation if the input ends with a question mark or the user pauses
    return input_buffer.strip().endswith("?") or len(input_buffer) > 50  # Adjust as needed

def handle_backspace(input_buffer, previous_input_ids, past_key_values):
    # If the input buffer is shorter than before, reset the KV cache
    if len(input_buffer) < len(previous_input_ids):
        past_key_values = None
    return past_key_values

# Define the request body schema using Pydantic
class StreamRequest(BaseModel):
    input_text: str

# Thread pool for concurrent requests
executor = ThreadPoolExecutor(max_workers=4)

# FastAPI app
web_app = FastAPI()

# Configure CORS
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (replace with your frontend URL in production)
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],  # Allow POST and OPTIONS methods
    allow_headers=["*"],  # Allow all headers
)

@web_app.post("/stream")
async def stream(request: StreamRequest):
    global past_key_values, input_buffer, previous_input_ids

    # Validate the input
    if not request.input_text:
        raise HTTPException(status_code=422, detail="Input text cannot be empty")

    # Update the input buffer
    input_buffer = request.input_text

    # Handle backspacing or edits
    past_key_values = handle_backspace(input_buffer, previous_input_ids, past_key_values)

    # Tokenize incrementally
    input_ids = tokenize_incrementally(tokenizer, input_buffer, previous_input_ids)
    previous_input_ids = input_ids

    # Update the KV cache
    past_key_values = update_kv_cache(model, input_ids, past_key_values)

    # Generate output only if the context is sufficient
    if should_generate_output(input_buffer):
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=50, past_key_values=past_key_values, use_cache=True)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        output_text = "Waiting for more input..."

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    return {"output": output_text}

# Deploy the FastAPI app as a Modal web endpoint
@app.function(
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A100",  # Use an A100 GPU
    timeout=86400,  # Timeout in seconds (1 day)
    allow_concurrent_inputs=100,  # Allow concurrent requests
    container_idle_timeout=600,  # Increase idle timeout
    volumes={"/cache": cache_volume},  # Mount the cache volume
)
@modal.asgi_app()
def fastapi_app():
    global tokenizer, model

    # Load the model only once when the container starts
    if tokenizer is None or model is None:
        load_model()

    return web_app