import torch
import copy
import numpy as np
from transformers import AutoModelForCausalLM
from typing import Dict, List
import matplotlib.pyplot as plt

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

class ActivationMagnitudeAnalyzer:
    def __init__(self, model, sf_type):
        """
        Initialize the activation magnitude analyzer
        
        Args:
            model: Transformer model to analyze
            sf_type: Superfloat quantization object
        """
        self.model = model
        self.sf_type = sf_type
        self.original_activations = {}
        self.quantized_activations = {}
        
        # Hook to capture layer activations
        self.hooks = []
        
    def register_activation_hooks(self, model):
        """
        Register forward hooks to capture activations for each layer
        
        Args:
            model: Model to register hooks on
        """
        def hook_fn(module, input, output, layer_name):
            # Handle different output types
            if hasattr(output, 'last_hidden_state'):
                # For models returning BaseModelOutputWithPast or similar
                output = output.last_hidden_state
            elif isinstance(output, tuple):
                # Take the first tensor if output is a tuple
                output = output[0]
            
            # Ensure output is a tensor
            if not isinstance(output, torch.Tensor):
                print(f"Warning: Unexpected output type for {layer_name}: {type(output)}")
                return 0.0
            
            # Compute average magnitude of activations
            avg_magnitude = torch.abs(output).mean().item()
            return avg_magnitude
        
        def register_hook(module, prefix=''):
            for name, child_module in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Add hook for specific layer types or all layers
                hook = child_module.register_forward_hook(
                    lambda module, input, output, name=full_name: 
                    self.original_activations.update({name: hook_fn(module, input, output, name)})
                )
                self.hooks.append(hook)
                
                # Recursively register hooks for nested modules
                register_hook(child_module, full_name)
        
        register_hook(model)
    
    def analyze_activations(self, input_data):
        """
        Analyze activation magnitudes for original and quantized models
        
        Args:
            input_data: Sample input for the model
        """
        # Clear previous hooks and activations
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.original_activations.clear()
        self.quantized_activations.clear()
        
        # Original model analysis
        self.register_activation_hooks(self.model)
        with torch.no_grad():
            _ = self.model(input_data)
        original_dict = self.original_activations.copy()
        
        # Quantize model (deep copy approach)
        quantized_model = self.quantize_model(copy.deepcopy(self.model))
        
        # Quantized model analysis
        self.register_activation_hooks(quantized_model)
        with torch.no_grad():
            _ = quantized_model(input_data)
        quantized_dict = self.original_activations.copy()
        
        self.original_activations = original_dict
        self.quantized_activations = quantized_dict
    
    def quantize_model(self, model):
        """
        Quantize the model using Superfloat
        
        Args:
            model: Model to quantize
        
        Returns:
            Quantized model
        """
        # Ensure the model is on the same device and float type
        model = model.to(self.sf_type.float_type).to(self.model.device)
        
        for name, param in model.named_parameters():
            with torch.no_grad():
                quantized_param = self.sf_type.tensor_quantize(param)
                param.data.copy_(quantized_param)
        return model
    
    # ... [rest of the previous implementation remains the same]

def main():
    # Superfloat setup
    sf = Superfloat(8)  # 8-bit quantization

    # Model and device setup
    model_name = "Qwen/Qwen2-0.5B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir='./', 
        token='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'
    )
    model = model.to(sf.float_type).to(device)

    # Prepare input data
    input_data = torch.randint(0, model.config.vocab_size, (1, 128), device=device)

    # Initialize and run analyzer
    analyzer = ActivationMagnitudeAnalyzer(model, sf)
    analyzer.analyze_activations(input_data)

    # Compare activations
    magnitude_diff = analyzer.compare_activations()

    # Cluster layers
    clustered_layers = analyzer.cluster_layers_by_deviation(magnitude_diff)

    # Visualize deviations
    analyzer.visualize_activation_deviations(magnitude_diff)

    # Print detailed results
    print("\nLayer Activation Magnitude Differences:")
    for layer, stats in magnitude_diff.items():
        print(f"{layer}: Original={stats['original']:.4f}, "
              f"Quantized={stats['quantized']:.4f}, "
              f"Relative Change={stats['relative_change_percent']:.2f}%")

    print("\nClustered Layers:")
    for cluster, layers in clustered_layers.items():
        print(f"{cluster.replace('_', ' ').title()}: {len(layers)} layers")

if __name__ == "__main__":
    main()