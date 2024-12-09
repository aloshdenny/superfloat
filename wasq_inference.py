import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

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
        decoded_value = (mantissa.to(torch.bfloat16) / (2**self.mantissa_bits - 1)) * self.max_val
        return decoded_value * (2 * sign - 1)  # Apply the sign

    def tensor_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantizes a tensor to the superfloat format, preserving the tensor's shape."""
        # Apply element-wise encoding to the entire tensor and then decode back
        encoded_tensor = self.encode(tensor)
        decoded_tensor = self.decode(encoded_tensor)
        return decoded_tensor

# Initialize Superfloat quantizer for clamping
sf = Superfloat(8)

# Load model in bfloat16 directly for inference
model_name = "meta-llama/Llama-3.2-1B"
model = LlamaForCausalLM.from_pretrained(model_name, cache_dir='./', token='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')
model.load_state_dict(torch.load("sf{sf.bits}_trained_epoch3", map_location=device))
model = model.to(torch.bfloat16).to(device)
model.eval()  # Ensure model is in inference mode

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

# Example usage
prompt = "It could be one of those nights, where we don't turn off the lights."
generated_text = quantized_inference(model, tokenizer, prompt)
print("Generated text:", generated_text)