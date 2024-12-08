import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained model and tokenizer
model_name = "Qwen/Qwen2-0.5B"

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./cache")
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache")
tokenizer.pad_token = tokenizer.eos_token

def prepare_hellaswag_dataset(tokenizer, max_length=128, num_samples=None):
    """
    Loads and tokenizes the HellaSwag dataset.
    
    Args:
        tokenizer: Tokenizer to use for preparing the dataset.
        max_length: Maximum token sequence length.
        num_samples: Number of samples to select from the dataset.
        
    Returns:
        A tokenized dataset ready for processing.
    """
    # Load the HellaSwag dataset (validation split)
    dataset = load_dataset("hellaswag", split="validation")
    
    # Optionally, select a subset of the dataset
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    # Tokenize each example
    def tokenize_example(example):
        return tokenizer(
            example['ctx'],  # Use the context field from the dataset
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

    # Apply tokenization to all examples
    tokenized_dataset = [tokenize_example(entry) for entry in tqdm(dataset, desc="Tokenizing HellaSwag")]

    return tokenized_dataset

def compute_layer_activation_magnitudes(model, tokenized_dataset):
    """
    Computes the average activation magnitude for each layer across all samples.
    
    Args:
        model: Pre-trained model.
        tokenized_dataset: Tokenized input dataset.
        
    Returns:
        A list of average activation magnitudes for each layer.
    """
    layer_activations = []

    def register_activation_hook(layers):
        def hook(module, input, output):
            # Capture the mean absolute activation magnitude
            if isinstance(output, torch.Tensor):
                layers.append(torch.abs(output).mean().item())
        return hook

    for tokenized_input in tqdm(tokenized_dataset, desc="Processing Samples"):
        # Reset activations for the current sample
        current_sample_layers = []
        
        # Register hooks for transformer layers
        hooks = []
        for name, module in model.named_modules():
            if 'transformer.layers' in name or 'model.layers' in name:  # Identify transformer layers
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

    # Compute the average activation magnitude across all samples
    avg_layer_activations = np.mean(layer_activations, axis=0)
    
    return avg_layer_activations

def visualize_layer_activation_magnitudes(avg_layer_activations):
    """
    Creates a bar graph of average layer activation magnitudes.
    
    Args:
        avg_layer_activations: List of average activation magnitudes per layer.
    """
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(avg_layer_activations)), avg_layer_activations, color='b', alpha=0.6)
    plt.title("Average Activation Magnitude Across Layers")
    plt.xlabel("Layer Index")
    plt.ylabel("Average Activation Magnitude")
    plt.xticks(range(len(avg_layer_activations)), [f"Layer {i+1}" for i in range(len(avg_layer_activations))])
    plt.tight_layout()
    plt.show()

def extract_inter_layer_attention_maps(model, tokenized_dataset):
    """
    Extracts inter-layer attention maps for all layers across samples.
    
    Args:
        model: Pre-trained model.
        tokenized_dataset: Tokenized input dataset.
        
    Returns:
        A list of attention maps for all layers and samples.
    """
    attention_maps = []

    def register_attention_hook(attention_outputs):
        def hook(module, inputs, outputs):
            # Store attention weights
            if isinstance(outputs, tuple):
                attention_outputs.append(outputs[0].detach().cpu())  # Attention maps often at index 0
        return hook

    for tokenized_input in tqdm(tokenized_dataset, desc="Processing Samples for Attention Maps"):
        current_sample_attentions = []
        
        # Register hooks for attention layers
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower():  # Identify attention layers
                hook = module.register_forward_hook(register_attention_hook(current_sample_attentions))
                hooks.append(hook)
        
        # Perform forward pass
        with torch.no_grad():
            input_ids = tokenized_input['input_ids'].squeeze(0).to(device)
            attention_mask = tokenized_input['attention_mask'].squeeze(0).to(device)
            model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Store attention maps for this sample
        attention_maps.append(current_sample_attentions)

    return attention_maps

def visualize_attention_heatmaps(attention_maps, tokenized_input, layer_idx=0):
    """
    Visualizes attention heatmaps for a specific layer.
    
    Args:
        attention_maps: List of attention maps for all layers.
        tokenized_input: Tokenized input for a specific sample.
        layer_idx: Index of the layer to visualize.
    """
    input_tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'].squeeze(0).tolist())
    attention_map = attention_maps[0][layer_idx].squeeze(0)  # [Batch, Heads, Tokens, Tokens]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_map.mean(dim=0),  # Average across heads
        xticklabels=input_tokens,
        yticklabels=input_tokens,
        cmap="viridis"
    )
    plt.title(f"Attention Map for Layer {layer_idx + 1}")
    plt.xlabel("Input Tokens")
    plt.ylabel("Output Tokens")
    plt.show()

# Prepare the dataset
tokenized_dataset = prepare_hellaswag_dataset(tokenizer, num_samples=50)

# Compute and visualize activation magnitudes
avg_layer_activations = compute_layer_activation_magnitudes(model, tokenized_dataset)
visualize_layer_activation_magnitudes(avg_layer_activations)

# Extract and visualize attention maps
attention_maps = extract_inter_layer_attention_maps(model, tokenized_dataset)
for i, tokenized_input in enumerate(tokenized_dataset[:5]):  # Limit to first 5 samples for visualization
    print(f"Sample {i + 1}: Visualizing attention heatmaps...")
    visualize_attention_heatmaps(attention_maps[i], tokenized_input, layer_idx=0)  # Visualize Layer 1