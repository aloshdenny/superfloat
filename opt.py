import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
import gc

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
        self.max_val = 2**(self.mantissa_bits - 1) - 1  # Adjusted max value calculation
        self.float_type = self.CASTING_TABLE[bits]

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        """Encode tensor to Superfloat representation."""
        # Normalize value to the range of representable values
        scaled = torch.clamp(value, min=-self.max_val, max=self.max_val)
        
        # Convert to fixed-point representation
        scaled_int = torch.round(scaled).to(torch.int32)
        
        # Encode sign bit
        sign_bit = (scaled_int < 0).to(torch.int32)
        abs_val = torch.abs(scaled_int)
        
        # Combine sign and magnitude
        encoded = (abs_val | (sign_bit << (self.mantissa_bits)))
        
        return encoded

    def decode(self, encoded_value: torch.Tensor) -> torch.Tensor:
        """Decode Superfloat representation back to floating-point."""
        # Extract magnitude and sign
        magnitude = encoded_value & ((1 << self.mantissa_bits) - 1)
        sign_bit = (encoded_value >> self.mantissa_bits) & 1
        
        # Convert back to signed integer
        signed_val = magnitude.to(self.float_type)
        signed_val = signed_val * (2 * sign_bit - 1)
        
        return signed_val

    def tensor_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor using Superfloat encoding and decoding."""
        # Encode to Superfloat 
        encoded = self.encode(tensor)
        
        # Decode back to floating point
        decoded = self.decode(encoded)
        
        # Track out-of-range values
        out_of_range = (tensor.abs() > self.max_val)
        
        return decoded, out_of_range

class QuantizedLlamaModel(nn.Module):
    def __init__(self, base_model: nn.Module, sf_quantizer: Superfloat):
        super().__init__()
        self.base_model = base_model
        self.sf_quantizer = sf_quantizer
        self.apply_quantization_hooks()

    def apply_quantization_hooks(self):
        """Apply Superfloat quantization hooks to model parameters."""
        for name, param in self.base_model.named_parameters():
            if 'weight' in name or 'bias' in name:
                # In-place quantization of parameters
                quantized_param, _ = self.sf_quantizer.tensor_quantize(param.data)
                param.data.copy_(quantized_param)

    def forward(self, x):
        """Quantize inputs and intermediate activations during forward pass."""
        # Quantize input
        x, _ = self.sf_quantizer.tensor_quantize(x)

        # Quantize each layer's computations
        for layer in self.base_model.children():
            if isinstance(layer, nn.Linear):
                # Quantize layer weights
                layer.weight.data, _ = self.sf_quantizer.tensor_quantize(layer.weight.data)
                
                # If layer has bias, quantize bias
                if layer.bias is not None:
                    layer.bias.data, _ = self.sf_quantizer.tensor_quantize(layer.bias.data)
            
            # Forward pass through layer
            x = layer(x)
            
            # Quantize activations
            x, _ = self.sf_quantizer.tensor_quantize(x)
        
        return x

# Main script remains largely the same, but use the improved QuantizedLlamaModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Superfloat configuration
sf = Superfloat(bits=11)  # 11-bit Superfloat 

# Load model
model_name = "meta-llama/Llama-3.2-1B"
model = LlamaForCausalLM.from_pretrained(model_name, cache_dir='./', token='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')
model = model.to(sf.float_type).to(device)

# Wrap model with Quantized Wrapper
quantized_model = QuantizedLlamaModel(model, sf)