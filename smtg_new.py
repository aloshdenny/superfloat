import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import copy

class WASQ:
    """
    Weight-Aware Selective Quantization (WASQ) class
    """
    def __init__(self, model, bits=11, activation_threshold=0.75):
        """
        Initialize WASQ with model and quantization parameters
        
        Args:
            model: Base model to quantize
            bits: Number of bits for quantization
            activation_threshold: Percentile of activation magnitude to target
        """
        self.original_model = model
        self.bits = bits
        self.activation_threshold = activation_threshold
        
        # Quantization casting table
        self.CASTING_TABLE = {
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
        self.float_type = self.CASTING_TABLE[bits]
        
        # Prepare for selective quantization
        self.layer_activations = None
        self.targeted_layers = None

    def compute_layer_activation_magnitudes(self, tokenized_dataset):
        """
        Compute activation magnitudes for each layer across samples
        """
        layer_activations = []

        def register_activation_hook(layers):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    layers.append(torch.abs(output).mean().item())
            return hook

        for tokenized_input in tqdm(tokenized_dataset, desc="Processing Samples"):
            current_sample_layers = []
            hooks = []
            
            for name, module in self.original_model.named_modules():
                if 'transformer.layers' in name or 'model.layers' in name:
                    hook = module.register_forward_hook(register_activation_hook(current_sample_layers))
                    hooks.append(hook)
            
            with torch.no_grad():
                input_ids = tokenized_input['input_ids'].squeeze(0)
                attention_mask = tokenized_input['attention_mask'].squeeze(0)
                self.original_model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
            
            for hook in hooks:
                hook.remove()
            
            layer_activations.append(current_sample_layers)

        # Compute average across all samples
        self.layer_activations = np.mean(layer_activations, axis=0)
        return self.layer_activations

    def select_targeted_layers(self):
        """
        Select layers for targeted quantization based on activation magnitude
        """
        if self.layer_activations is None:
            raise ValueError("Layer activations not computed. Run compute_layer_activation_magnitudes first.")
        
        # Calculate activation threshold
        threshold = np.percentile(self.layer_activations, 
                                  (1 - self.activation_threshold) * 100)
        
        # Select layers above the threshold
        self.targeted_layers = np.where(self.layer_activations >= threshold)[0]
        
        return self.targeted_layers

    def visualize_layer_activations(self):
        """
        Visualize layer activation magnitudes
        """
        if self.layer_activations is None:
            raise ValueError("Layer activations not computed.")
        
        plt.figure(figsize=(15, 6))
        plt.bar(range(len(self.layer_activations)), self.layer_activations)
        plt.title('Layer Activation Magnitudes')
        plt.xlabel('Layer Index')
        plt.ylabel('Activation Magnitude')
        plt.xticks(range(len(self.layer_activations)), 
                   [f'Layer {i+1}' for i in range(len(self.layer_activations))])
        
        # Highlight targeted layers
        if self.targeted_layers is not None:
            for layer in self.targeted_layers:
                plt.bar(layer, self.layer_activations[layer], color='red', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

    def quantize_model(self, model):
        """
        Selectively quantize model layers based on activation magnitudes
        """
        quantized_model = copy.deepcopy(model)
        
        for name, param in quantized_model.named_parameters():
            # Extract layer index from parameter name
            try:
                layer_idx = int(name.split('.')[2])
            except:
                continue
            
            # Quantize only targeted layers
            if layer_idx in self.targeted_layers:
                # Implement your quantization logic here
                param.data = param.data.to(self.float_type)
                # You can add more sophisticated quantization here
        
        return quantized_model

def prepare_dataset(tokenizer, max_length=128, num_samples=100):
    """
    Prepare dataset for analysis
    """
    dataset = load_dataset("hellaswag", split="validation")
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    def tokenize_example(example):
        return tokenizer(
            example['ctx'],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

    tokenized_dataset = [tokenize_example(entry) for entry in tqdm(dataset, desc="Tokenizing HellaSwag")]

    return tokenized_dataset

def main():
    # Initialize model and tokenizer
    model_name = "Qwen/Qwen2-0.5B"
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./cache", token='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')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache", token='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')
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    tokenized_dataset = prepare_dataset(tokenizer)

    # Initialize WASQ
    wasq = WASQ(model, bits=11, activation_threshold=0.75)

    # Compute layer activations
    layer_activations = wasq.compute_layer_activation_magnitudes(tokenized_dataset)

    # Select targeted layers
    targeted_layers = wasq.select_targeted_layers()
    print("Targeted Layers:", targeted_layers)

    # Visualize layer activations
    wasq.visualize_layer_activations()

    # Quantize model
    quantized_model = wasq.quantize_model(model)

    return quantized_model

if __name__ == "__main__":
    main()