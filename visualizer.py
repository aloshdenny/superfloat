import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset, Dataset

# Load model and tokenizer
model_name = "Qwen/Qwen2-0.5B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='./', token='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')
model = model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./', token='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')
tokenizer.pad_token = tokenizer.eos_token

# Register gradient hooks to capture gradients
def register_gradient_hooks(model):
    gradients = {}

    def save_gradients(name):
        def hook(grad):
            gradients[name] = grad.clone().detach()
        return hook

    for name, param in model.named_parameters():
        param.register_hook(save_gradients(name))

    return gradients

# Plot gradient magnitudes for visualization
def plot_gradient_flow(gradients):
    grad_magnitudes = {name: grad.norm().item() for name, grad in gradients.items()}
    plt.figure(figsize=(12, 6))
    plt.bar(grad_magnitudes.keys(), grad_magnitudes.values(), color='b', alpha=0.6)
    plt.xlabel("Parameter Name", fontsize=12)
    plt.ylabel("Gradient Magnitude", fontsize=12)
    plt.title("Gradient Flow Analysis", fontsize=16)
    plt.xticks(rotation=90, fontsize=10)
    plt.tight_layout()
    plt.show()

def extract_attention_maps(model, inputs):
    attention_maps = []

    def hook(module, inputs, outputs):
        attention_maps.append(outputs[1].detach())  # Ensure we're saving attention weights

    # Register hooks on all attention layers
    hooks = []
    for name, module in model.named_modules():
        if "attention" in name.lower():  # Identify attention layers
            hooks.append(module.register_forward_hook(hook))

    # Perform forward pass and make sure we request attention outputs
    with torch.no_grad():
        model(**inputs, output_attentions=True)

    # Remove hooks after forward pass
    for hook in hooks:
        hook.remove()

    return attention_maps

def visualize_attention(attention_map, input_tokens, layer_idx=0, head_idx=0):
    # Extract the attention map for the specific layer
    attention_map_data = attention_map[layer_idx]  # Extract the specific layer's attention weights

    # Check the shape of the attention map to ensure it's 3D
    print(f"Attention map shape for layer {layer_idx}: {attention_map_data.shape}")

    # Assuming the attention map shape is [batch_size, num_heads, seq_len, seq_len]
    # Select the first sample (batch_size index 0) and the head at head_idx
    if len(attention_map_data.shape) == 3:
        attention_map_data = attention_map_data[0, head_idx]  # Select batch (0) and head (head_idx)

    # Ensure the attention map is 2D (seq_len x seq_len)
    if len(attention_map_data.shape) == 2:
        print("Correct shape for heatmap")
    else:
        print(f"Warning: Attention map data does not have the correct shape for visualization. Found shape: {attention_map_data.shape}")
        return

    # Now pass the attention map data to seaborn
    sns.heatmap(
        attention_map_data,
        xticklabels=input_tokens,
        yticklabels=input_tokens,
        cmap="magma"
    )
    plt.title(f"Layer {layer_idx + 1}, Head {head_idx + 1} Attention Map", fontsize=14)
    plt.xlabel("Input Tokens")
    plt.ylabel("Output Tokens")
    plt.show()

# Tokenize a batch of test inputs
def tokenize_inputs(texts, tokenizer, max_length=128):
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return inputs

# Function to load and tokenize the HellaSwag dataset
def prepare_hellaswag_dataset(tokenizer, max_length=128):
    """
    Loads the HellaSwag dataset and tokenizes it for the model.
    
    Args:
        tokenizer: The tokenizer to use for tokenizing text.
        max_length: Maximum length for tokenized sequences.
    
    Returns:
        Tokenized dataset as a list of dictionaries.
    """
    # Load validation split of HellaSwag
    dataset = load_dataset("hellaswag", split="validation")

    # Tokenize prompts (contexts)
    def tokenize_example(example):
        return tokenizer(
            example['ctx'],  # Context field in HellaSwag
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

    # Tokenize all examples in the dataset
    tokenized_dataset = [tokenize_example(entry) for entry in tqdm(dataset, desc="Tokenizing HellaSwag")]

    return tokenized_dataset

# Load the tokenized HellaSwag dataset
tokenized_dataset = prepare_hellaswag_dataset(tokenizer)

def analyze_hellaswag_with_gradient_and_attention(model, tokenizer, tokenized_dataset, num_samples=5):
    """
    Performs gradient flow analysis and visualizes attention maps for the HellaSwag dataset.
    
    Args:
        model: The pretrained model.
        tokenizer: The tokenizer.
        tokenized_dataset: Tokenized HellaSwag dataset.
        num_samples: Number of samples to analyze.
    """
    # Register gradient hooks for gradient flow analysis
    gradients = register_gradient_hooks(model)

    # Process a subset of the dataset
    for idx, tokenized_input in enumerate(tokenized_dataset[:num_samples]):
        input_ids = tokenized_input['input_ids'].squeeze(0).to(device)
        attention_mask = tokenized_input['attention_mask'].squeeze(0).to(device)

        # Forward pass with output_attentions=True to extract attention weights
        outputs = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), output_attentions=True)
        logits = outputs.logits
        attention_weights = outputs.attentions  # List of attention layers

        # Compute dummy loss (shifted target for language modeling task)
        target = input_ids[1:]  # Shift input as target
        logits = logits[:, :-1].contiguous()
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))

        # Backward pass for gradient flow
        loss.backward()

        # Visualize Gradient Flow
        print(f"\nGradient Flow Analysis for Sample {idx + 1}")
        plot_gradient_flow(gradients)

        # Extract and visualize attention maps
        print(f"Attention Map Visualization for Sample {idx + 1}")
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu().tolist())
        
        # Choose a layer and head to visualize (e.g., layer 0, head 0)
        layer_idx = 0
        head_idx = 0

        # Attention weights are a list of (batch_size, num_heads, seq_len, seq_len) tensors for each layer
        attention_map = attention_weights[layer_idx][0, head_idx].cpu().detach().numpy()
        
        # Visualize the attention map
        visualize_attention(attention_map, input_tokens, layer_idx=layer_idx, head_idx=head_idx)

        # Clear gradients for the next sample
        model.zero_grad()

analyze_hellaswag_with_gradient_and_attention(model, tokenizer, tokenized_dataset, num_samples=5)