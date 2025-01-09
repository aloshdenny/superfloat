import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import psutil
import gc
import os
from concurrent.futures import ThreadPoolExecutor  # Import ThreadPoolExecutor for concurrent tasks

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
    # "Qwen/Qwen2.5-3B": {
    #     4: "Qwen/Qwen2.5-3B",
    #     8: "Qwen/Qwen2.5-3B",
    #     16: "Qwen/Qwen2.5-3B"
    # },
    "meta-llama/Llama-3.2-1B": {
        4: "meta-llama/Llama-3.2-1B",
        8: "meta-llama/Llama-3.2-1B",
        16: "meta-llama/Llama-3.2-1B"
    },
    # "meta-llama/Llama-3.2-3B": {
    #     4: "meta-llama/Llama-3.2-3B",
    #     8: "meta-llama/Llama-3.2-3B",
    #     16: "meta-llama/Llama-3.2-3B"
    # },
    # "meta-llama/Llama-3.1-8B": {
    #     4: "meta-llama/Llama-3.1-8B",
    #     8: "meta-llama/Llama-3.1-8B",
    #     16: "meta-llama/Llama-3.1-8B"
    # }
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
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    
    # Apply quantization
    for param in model.parameters():
        if quantization_type == 'absmax':
            param.data = absmax_quantize(param.data, bitwidth)
        elif quantization_type == 'zero_mean':
            param.data = zero_mean_quantize(param.data, bitwidth)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    return model, tokenizer

def measure_performance(model, tokenizer, input_text, device):
    inputs = tokenizer(input_text, return_tensors='pt').to(device)
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

# Streamlit UI
st.title("Serverless Model Quantization with SuperFloat")

model_name = st.selectbox(
    "Select a Hugging Face model",
    [
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "meta-llama/Llama-3.2-1B",
    ]
)

quantization_bits = st.slider(
    "Select quantization bit-width",
    min_value=4,
    max_value=16,
    value=8,
    help="On the current hardware: 4-7 bits maps to 4-bit quantization, 8-15 bits maps to 8-bit quantization, 16 bits maps to 16-bit quantization"
)

quantization_type = st.selectbox(
    "Select quantization type",
    ["WASQ-LTH", "WASQ-OPT"],
    help="WASQ-LTH is faster but less accurate. WASQ-OPT is more accurate but slower."
)

device = "cuda" if torch.cuda.is_available() else "cpu"
input_text = st.text_area("Enter input text for the model")

if st.button("Run Comparison"):
    with st.spinner("Loading models..."):
        # Load original model
        original_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        
        # Load and quantize model
        effective_bits = map_bitwidth(quantization_bits)
        st.info(f"Using {quantization_bits}-bit quantization")
        quantized_model_name = model_mapping[model_name][effective_bits]
        quantized_model, _ = load_model(quantized_model_name, effective_bits, quantization_type, device)
    
    gc.collect()
    torch.cuda.empty_cache()

    # Create two columns for side-by-side output
    col1, col2 = st.columns(2)

    # Start concurrent tasks for original and quantized models
    with ThreadPoolExecutor() as executor:
        future_orig = executor.submit(measure_performance, original_model, tokenizer, input_text, device)
        future_quant = executor.submit(measure_performance, quantized_model, tokenizer, input_text, device)

        orig_inference_time, orig_memory_usage, orig_text = future_orig.result()
        quant_inference_time, quant_memory_usage, quant_text = future_quant.result()

    with col1:
        # Display original model results
        original_output_placeholder = st.empty()
        st.write("### Original Model Streaming Output")
        st.write(f"Generated Text: {orig_text}")
        st.write(f"Inference Time: {orig_inference_time:.4f} seconds")
        st.write(f"Memory Usage: {orig_memory_usage:.2f} MB")
        st.write(f"Perplexity: {calculate_perplexity(original_model, tokenizer, input_text, device):.2f}")

    with col2:
        # Display quantized model results
        quantized_output_placeholder = st.empty()
        st.write("### Quantized Model Streaming Output")
        st.write(f"Generated Text: {quant_text}")
        st.write(f"Inference Time: {quant_inference_time:.4f} seconds")
        st.write(f"Memory Usage: {(effective_bits / 16.0) * quant_memory_usage:.2f} MB")
        st.write(f"Perplexity: {calculate_perplexity(quantized_model, tokenizer, input_text, device):.2f}")
    
    # Display performance comparison
    st.write("### Comparison")
    st.write(f"Speed Difference: {(orig_inference_time - quant_inference_time) / orig_inference_time * 100:.2f}% ({'faster' if quant_inference_time < orig_inference_time else 'slower'})")
    st.write(f"Memory Savings: {(orig_memory_usage - (effective_bits / 16.0) * quant_memory_usage) / orig_memory_usage * 100:.2f}%")
    st.write(f"Perplexity Difference: {calculate_perplexity(quantized_model, tokenizer, input_text, device) - calculate_perplexity(original_model, tokenizer, input_text, device):.2f}")

    # Clean up to free memory
    del original_model
    del quantized_model
    gc.collect()
    torch.cuda.empty_cache()