import torch
from transformers import AutoModelForCausalLM

class Superfloat:
    CASTING_TABLE = {
        16: torch.float32,
        15: torch.float32,
        14: torch.float32,
        13: torch.float32,
        12: torch.float32,
        11: torch.float16,
        10: torch.float16,
        9: torch.float16,
        8: torch.bfloat16,
        7: torch.bfloat16,
        6: torch.bfloat16,
        5: torch.bfloat16,
        4: torch.bfloat16,
    }

    def __init__(self, bits: int):
        assert 4 <= bits <= 16, "Superfloat bitwidth must be between 4 and 16."
        self.bits = bits
        self.mantissa_bits = bits - 1
        self.max_val = 1 - 2**-self.mantissa_bits  # Precompute max representable value
        self.float_type = self.CASTING_TABLE[bits]  # Get float type based on bitwidth

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
        decoded_value = (mantissa.to(self.float_type) / (2**self.mantissa_bits - 1)) * self.max_val
        return decoded_value * (2 * sign - 1)

    def tensor_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantizes a tensor to the superfloat format and decodes back."""
        encoded_tensor = self.encode(tensor)
        decoded_tensor = self.decode(encoded_tensor)
        return decoded_tensor

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
        if param_data.dtype != sf_type.float_type:
            print(f"Parameter {name} is not in {sf_type.float_type} format!")
            all_parameters_valid = False
        if not torch.all((param_data >= -sf_type.max_val) & (param_data <= sf_type.max_val)):
            print(f"Parameter {name} has values outside the SF{sf_type.bits} range!")
            all_parameters_valid = False
    return all_parameters_valid

# Load model
model_name = "Qwen/Qwen2-0.5B"
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='./', token='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')
model = model.to(sf.float_type).to(device)  # Ensure model uses the correct float type

# Quantize parameters
print("Quantizing model parameters...")
quantized_model = quantize_model(model, sf)

# Verify quantization
print("Checking model quantization...")
is_valid = check_model_quantization(quantized_model, sf)

if is_valid:
    print(f"All parameters are in {sf.float_type} format and within the SF{sf.bits} range.")
    # Save quantized model
    save_path = f"sf{sf.bits}_vanilla"
    torch.save(quantized_model.state_dict(), save_path)
    print(f"Quantized model saved to {save_path}")
else:
    print("Model quantization verification failed. Fix issues before saving.")