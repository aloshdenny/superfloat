import modal

# Create a Modal image with the required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "huggingface_hub",
    )
    .apt_install("gcc", "python3-dev")  # Add necessary system libraries if needed
)

app = modal.App("test")

@app.function(gpu="H100", image=image, timeout=86400)
def train_and_upload():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm
    from datasets import load_dataset

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class Superfloat:
        def __init__(self, bits: int):
            assert 2 <= bits <= 16, "Superfloat bitwidth must be between 4 and 16."
            self.bits = bits
            self.mantissa_bits = bits - 1
            self.max_val = 1 - 2**-self.mantissa_bits  # Precompute max representable value

        def encode(self, value: torch.Tensor) -> torch.Tensor:
            """Encodes a tensor of values into the superfloat format."""
            clipped_value = torch.clamp(value, min=-self.max_val, max=self.max_val)
            mantissa = (torch.abs(clipped_value) * (2**self.mantissa_bits - 1) / self.max_val).floor().to(torch.int32)
            sign = (clipped_value < 0).to(torch.int32)
            return (mantissa | (sign << self.mantissa_bits)).to(torch.int32)

        def decode(self, encoded_value: torch.Tensor) -> torch.Tensor:
            """Decodes a tensor of encoded superfloat values to regular floats."""
            mantissa = encoded_value & ((1 << self.mantissa_bits) - 1)
            sign = (encoded_value >> self.mantissa_bits) & 1
            decoded_value = (mantissa.to(torch.float32) / (2**self.mantissa_bits - 1)) * self.max_val
            return decoded_value * (2 * sign - 1)

        def handle_outliers(self, tensor: torch.Tensor, percentile: float = 99.9) -> torch.Tensor:
            """Clips outliers based on percentile threshold."""
            threshold = torch.quantile(torch.abs(tensor), percentile/100)
            return torch.clamp(tensor, -threshold, threshold)
        
        def quantize_per_channel(self, tensor: torch.Tensor) -> torch.Tensor:
            """Applies per-channel quantization for weight matrices."""
            original_shape = tensor.shape
            
            # Reshape to [out_channels, -1]
            tensor_2d = tensor.reshape(original_shape[0], -1)
            
            # Compute scales per channel
            scales = torch.max(torch.abs(tensor_2d), dim=1, keepdim=True)[0]
            scales = torch.clamp(scales, min=1e-5)  # Prevent division by zero
            
            # Normalize, quantize, and rescale each channel
            normalized = tensor_2d / scales
            encoded = self.encode(normalized)
            decoded = self.decode(encoded)
            rescaled = decoded * scales
            
            # Reshape back to original shape
            return rescaled.reshape(original_shape)
        
        def quantize_attention_weights(self, tensor: torch.Tensor) -> torch.Tensor:
            """Special handling for attention weights using zero-mean normalization."""
            mean = torch.mean(tensor)
            centered = tensor - mean
            scale = torch.max(torch.abs(centered))
            scale = torch.clamp(scale, min=1e-5)
            
            normalized = centered / scale
            encoded = self.encode(normalized)
            decoded = self.decode(encoded)
            
            return (decoded * scale) + mean
        
        def quantize_layernorm_params(self, tensor: torch.Tensor) -> torch.Tensor:
            """Special handling for LayerNorm parameters with higher precision."""
            # For LayerNorm, we use a more conservative quantization
            scale = torch.max(torch.abs(tensor))
            scale = torch.clamp(scale, min=1e-5)
            
            normalized = tensor / scale
            encoded = self.encode(normalized)
            decoded = self.decode(encoded)
            
            return decoded * scale
        
        def tensor_quantize(self, tensor: torch.Tensor, layer_type: str = "default") -> torch.Tensor:
            """Enhanced quantization with different strategies based on layer type."""
            # Handle outliers first
            tensor = self.handle_outliers(tensor)
            
            if layer_type == "attention":
                # For attention layers, use zero-mean quantization
                return self.quantize_attention_weights(tensor)
            
            elif layer_type == "layernorm":
                # For LayerNorm parameters, use higher precision quantization
                return self.quantize_layernorm_params(tensor)
            
            elif len(tensor.shape) > 1:
                # For weight matrices (2D+), use per-channel quantization
                return self.quantize_per_channel(tensor)
            
            else:
                # For 1D tensors (biases, etc.), use simple absmax scaling
                scale = torch.max(torch.abs(tensor))
                scale = torch.clamp(scale, min=1e-5)
                normalized = tensor / scale
                encoded = self.encode(normalized)
                decoded = self.decode(encoded)
                return decoded * scale

    # Initialize Superfloat quantizer for sf{sf.bits} clamping
    sf = Superfloat(4)

    # Function to quantize the model
    def quantize_model(model, sf_type, zero_mean=False, absmax_scale=False):
        for name, param in model.named_parameters():
            print(f"Quantizing: {name}")
            with torch.no_grad():
                quantized_param = sf_type.tensor_quantize(param, zero_mean=zero_mean, absmax_scale=absmax_scale)
                param.data.copy_(quantized_param)
        return model


    # Checker function to verify quantization
    def check_model_quantization(model, sf_type):
        all_parameters_valid = True
        for name, param in model.named_parameters():
            param_data = param.data
            if not torch.all((param_data >= -sf_type.max_val) & (param_data <= sf_type.max_val)):
                print(f"Parameter {name} has values outside the SF{sf_type.bits} range!")
                all_parameters_valid = False
        return all_parameters_valid

    # Load model
    model_name = "meta-llama/Llama-3.2-1B"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./", token='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')
    tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare model
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./", token='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')
    model = model.to(device)
    model.eval()

    # Quantize parameters
    print("Quantizing model parameters...")
    quantized_model = quantize_model(model, sf)

    # Verify quantization
    print("Checking model quantization...")
    is_valid = check_model_quantization(quantized_model, sf)

    if is_valid:
        print(f"All parameters are within the SF{sf.bits} range.")
    else:
        print("Model quantization verification failed. Fix issues before proceeding.")

    # Function to calculate perplexity
    def calculate_perplexity(model, tokenizer, prompt):
        """Calculates the perplexity of the model on a given prompt."""
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # Get model outputs (logits)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

        # Get the loss (cross entropy) from the model's output
        loss = outputs.loss  # This is the cross-entropy loss

        # Compute perplexity: exp(loss)
        perplexity = torch.exp(loss)
        return perplexity.item()

    # Function to load the HellaSwag dataset
    def load_hellaswag_data():
        """Load the HellaSwag dataset from Hugging Face."""
        dataset = load_dataset("hellaswag", split='validation', trust_remote_code=True)

        # Extract only the prompts (contexts) for evaluation
        prompts = [entry['ctx'] for entry in dataset]

        # Return the prompts as a list
        return prompts

    # Load HellaSwag data (prompts)
    prompts = load_hellaswag_data()

    total_perplexity = 0.0
    for prompt in tqdm(prompts, desc="Calculating perplexity"):
        perplexity = calculate_perplexity(quantized_model, tokenizer, prompt)
        total_perplexity += perplexity

    print(total_perplexity/len(prompts))

    # Compute and print average perplexity
    average_perplexity = total_perplexity / len(prompts)
    print(f"Average Perplexity: {average_perplexity}")

@app.local_entrypoint()
def main():
    train_and_upload.remote()