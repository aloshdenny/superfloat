import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from tqdm import tqdm
from datasets import load_dataset

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_dir = "./"

# Function to load model
def load_model(model_path):
    model = LlamaForCausalLM.from_pretrained(model_name, cache_dir='./', token='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')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(torch.bfloat16).to(device)
    model.eval()  # Ensure model is in inference mode
    return model

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
        decoded_value = (mantissa.to(torch.bfloat16) / (2**self.mantissa_bits - 1)) * self.max_val
        return decoded_value * (2 * sign - 1)  # Apply the sign

    def tensor_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantizes a tensor to the superfloat format, preserving the tensor's shape."""
        # Apply element-wise encoding to the entire tensor and then decode back
        encoded_tensor = self.encode(tensor)
        decoded_tensor = self.decode(encoded_tensor)
        return decoded_tensor

# Initialize Superfloat quantizer for sf{sf.bits}amping
sf = Superfloat(8)

model_name = "meta-llama/Llama-3.2-1B"

# Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, cache_dir='./', token='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')
tokenizer.pad_token = tokenizer.eos_token

def quantized_inference(model, tokenizer, prompt, max_length=500):
    """Runs inference on a prompt with activation quantization using Superfloat."""
    # Encode input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        # Perform generation with clamped activations
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

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

# Model paths
import os

def get_model_paths(base_dir, sf_bits):
    """
    Dynamically generate model paths based on the sf.bits format.
    Looks for models of the form:
    1. sf{sf_bits}_vanilla
    2. sf{sf_bits}_{epoch_num}_fpm
    3. sf{sf_bits}_{epoch_num}_opt
    
    Args:
        base_dir (str): The directory where the models are stored.
        sf_bits (int): The bitwidth for the Superfloat quantizer.
    
    Returns:
        List of model paths.
    """
    model_paths = []
    model_pattern = f"sf{sf_bits}_"

    # Scan directory for models matching the pattern
    for model_name in os.listdir(base_dir):
        if model_name.startswith(model_pattern):
            model_paths.append(os.path.join(base_dir, model_name))

    # Ensure models are sorted to follow the desired order: vanilla -> fpm -> opt
    model_paths.sort()
    
    return model_paths

# Function to evaluate perplexity for a list of models and prompts
def evaluate_models(base_dir, sf_bits, tokenizer, prompts):
    """
    Evaluates models dynamically loaded based on the sf.bits format.
    
    Args:
        base_dir (str): The directory where the models are stored.
        sf_bits (int): The bitwidth for the Superfloat quantizer.
        tokenizer: The tokenizer to use for model inference.
        prompts: The list of prompts to evaluate.
    
    Returns:
        Dictionary with model names and their corresponding average perplexity.
    """
    model_perplexities = {}

    # Get dynamically generated model paths
    models = get_model_paths(base_dir, sf_bits)

    for model_path in models:
        model = load_model(model_path)
        print(f"Evaluating model: {model_path}")

        total_perplexity = 0.0
        num_prompts = len(prompts)

        # Compute perplexity for each prompt
        for prompt in tqdm(prompts, desc=f"Processing {model_path}", leave=False):
            perplexity = calculate_perplexity(model, tokenizer, prompt)
            total_perplexity += perplexity

        # Average perplexity for the current model
        avg_perplexity = total_perplexity / num_prompts
        model_perplexities[model_path] = avg_perplexity
        print(f"Average Perplexity for {model_path}: {avg_perplexity}")

    return model_perplexities

# Function to load the HellaSwag dataset
def load_hellaswag_data():
    """Load the HellaSwag dataset from Hugging Face."""
    dataset = load_dataset("hellaswag", split='validation')

    # Extract only the prompts (contexts) for evaluation
    prompts = [entry['ctx'] for entry in dataset]

    # Return the prompts as a list
    return prompts

# Load HellaSwag data (prompts)
prompts = load_hellaswag_data()

# Evaluate all models on HellaSwag prompts
model_perplexities = evaluate_models(base_dir, sf.bits, tokenizer, prompts)

# Print final results
print("\nAverage Perplexities for all models:")
for model_path, avg_perplexity in model_perplexities.items():
    print(f"{model_path}: {avg_perplexity}")