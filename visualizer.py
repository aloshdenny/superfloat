import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Load model and tokenizer
model_name = "Qwen/Qwen2-0.5B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def prepare_hellaswag_dataset(tokenizer, max_length=128, num_samples=None):
    """
    Loads and tokenizes the HellaSwag dataset.
    """
    dataset = load_dataset("hellaswag", split="validation")
    
    if num_samples is not None:
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

def compute_layer_activation_magnitudes(model, tokenized_dataset):
    """
    Compute activation magnitudes for each layer across all samples.
    """
    layer_activations = []

    def register_activation_hook(layers):
        def hook(module, input, output):
            # Capture magnitude of activations
            if isinstance(output, torch.Tensor):
                # Compute mean absolute activation magnitude
                layers.append(torch.abs(output).mean().item())
        return hook

    for tokenized_input in tqdm(tokenized_dataset, desc="Processing Samples"):
        # Reset layer activations for each sample
        current_sample_layers = []
        
        # Remove any existing hooks
        hooks = []
        
        # Register hooks on transformer layers
        for name, module in model.named_modules():
            if 'transformer.layers' in name or 'model.layers' in name:
                hook = module.register_forward_hook(register_activation_hook(current_sample_layers))
                hooks.append(hook)
        
        # Perform forward pass
        with torch.no_grad():
            input_ids = tokenized_input['input_ids'].squeeze(0).to(device)
            attention_mask = tokenized_input['attention_mask'].squeeze(0).to(device)
            model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Store layer activations for this sample
        layer_activations.append(current_sample_layers)

    # Compute average across all samples
    avg_layer_activations = np.mean(layer_activations, axis=0)
    
    return avg_layer_activations

def visualize_layer_activation_magnitudes(avg_layer_activations):
    """
    Create a bar graph of average layer activation magnitudes.
    """
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(avg_layer_activations)), avg_layer_activations)
    plt.title('Average Activation Magnitude Across Layers')
    plt.xlabel('Layer Index')
    plt.ylabel('Average Activation Magnitude')
    plt.xticks(range(len(avg_layer_activations)), [f'Layer {i+1}' for i in range(len(avg_layer_activations))])
    plt.tight_layout()
    plt.show()

# Prepare dataset
tokenized_dataset = prepare_hellaswag_dataset(tokenizer, num_samples=100)

# Compute and visualize layer activation magnitudes
avg_layer_activations = compute_layer_activation_magnitudes(model, tokenized_dataset)
visualize_layer_activation_magnitudes(avg_layer_activations)