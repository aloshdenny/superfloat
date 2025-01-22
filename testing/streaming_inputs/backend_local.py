from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from concurrent.futures import ThreadPoolExecutor
import gc

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the tokenizer and model
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

# Thread pool for concurrent requests
executor = ThreadPoolExecutor(max_workers=4)

# Global variables to store input buffer and KV cache
input_buffer = ""
past_key_values = None

def process_input(new_input):
    global input_buffer, past_key_values

    # Update the input buffer
    input_buffer = new_input

    # Tokenize the input
    inputs = tokenizer(input_buffer, return_tensors="pt").to("cuda")

    # Perform incremental inference with KV caching
    with torch.no_grad():
        outputs = model(inputs.input_ids, past_key_values=past_key_values)
        past_key_values = outputs.past_key_values

    # Decode the output tokens
    output_text = tokenizer.decode(outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True)

    return output_text

@app.route("/stream", methods=["POST"])
def stream():
    data = request.json
    new_input = data.get("input", "")

    # Submit the input processing task to the thread pool
    future = executor.submit(process_input, new_input)
    output_text = future.result()

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    return jsonify({"output": output_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)