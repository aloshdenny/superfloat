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

# Load the tokenizer and model inside the Modal container
def load_model():
    global tokenizer, model

    model_name = "meta-llama/Llama-3.2-3B"  # Example model, choose your preferred model
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
    global past_key_values, input_buffer

    # Validate the input
    if not request.input_text:
        raise HTTPException(status_code=422, detail="Input text cannot be empty")

    # Append incoming text to the input buffer
    input_buffer += request.input_text

    # Tokenize the entire input buffer (using subword-level or word-level)
    inputs = tokenizer(input_buffer, return_tensors="pt", add_special_tokens=True).to("cuda") 

    # Perform incremental inference with KV caching
    with torch.no_grad():
        outputs = model(inputs.input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values

    # Decode the generated tokens
    generated_token_ids = outputs.logits.argmax(dim=-1).squeeze().tolist()
    output_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True) 

    # Clear the buffer to avoid reprocessing (optional: keep some context for better generation)
    # input_buffer = "" 

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