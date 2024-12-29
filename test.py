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

    # Define Superfloat quantizer for clamping activations
    class Superfloat:
        def __init__(self, bits: int):
            assert 4 <= bits <= 16, "Superfloat bitwidth must be between 4 and 16."
            self.bits = bits
            self.mantissa_bits = bits - 1
            self.max_val = 1 - 2**-self.mantissa_bits  # Precompute max representable value

        def encode(self, value: torch.Tensor) -> torch.Tensor:
            """Encodes a tensor of values into the superfloat format with optimized operations."""
            # Clip tensor values to the valid range for SFx
            clipped_value = torch.clamp(value, min=-self.max_val, max=self.max_val)

            # Calculate mantissa representation element-wise
            mantissa = (torch.abs(clipped_value) * (2**self.mantissa_bits - 1) / self.max_val).floor().to(torch.int32)

            # Create the superfloat representation (1 bit for sign and mantissa bits)
            sign = (clipped_value < 0).to(torch.int32)
            return (mantissa | (sign << self.mantissa_bits)).to(torch.int32)

        def decode(self, encoded_value: torch.Tensor) -> torch.Tensor:
            """Decodes a tensor of encoded superfloat values to regular floats."""
            # Extract mantissa and sign from the encoded superfloat
            mantissa = encoded_value & ((1 << self.mantissa_bits) - 1)
            sign = (encoded_value >> self.mantissa_bits) & 1

            # Calculate the decoded float using the mantissa and max_val
            decoded_value = (mantissa.to(torch.float32) / (2**self.mantissa_bits - 1)) * self.max_val
            return decoded_value * (2 * sign - 1)  # Apply the sign

        def tensor_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
            """Quantizes a tensor to the superfloat format, preserving the tensor's shape."""
            # Apply element-wise encoding to the entire tensor and then decode back
            encoded_tensor = self.encode(tensor)
            decoded_tensor = self.decode(encoded_tensor)
            return decoded_tensor

    # Initialize Superfloat quantizer for sf{sf.bits} clamping
    sf = Superfloat(8)

    # Function to quantize the model
    def quantize_model(model, sf_type):
        for name, param in model.named_parameters():
            print(f"Quantizing: {name}")
            with torch.no_grad():
                quantized_param = sf_type.tensor_quantize(param)
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

    # Evaluate perplexity on a subset of prompts
    num_prompts = 100  # Limit to 100 prompts for evaluation
    selected_prompts = prompts[:num_prompts]

    total_perplexity = 0.0
    for prompt in tqdm(selected_prompts, desc="Calculating perplexity"):
        perplexity = calculate_perplexity(quantized_model, tokenizer, prompt)
        total_perplexity += perplexity

    print(total_perplexity)

    # Compute and print average perplexity
    average_perplexity = total_perplexity / len(selected_prompts)
    print(f"Average Perplexity: {average_perplexity}")

@app.local_entrypoint()
def main():
    train_and_upload.remote()