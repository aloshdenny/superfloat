import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

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
    Loads the HellaSwag dataset and tokenizes it for the model.
    
    Args:
        tokenizer: The tokenizer to use for tokenizing text.
        max_length: Maximum length for tokenized sequences.
        num_samples: Number of samples to use (None for full dataset)
    
    Returns:
        Tokenized dataset
    """
    # Load validation split of HellaSwag
    dataset = load_dataset("hellaswag", split="validation")
    
    # Limit samples if specified
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

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

def compute_average_attention_activations(model, tokenized_dataset):
    """
    Compute average attention activations across all samples.
    
    Args:
        model: The pretrained model
        tokenized_dataset: Tokenized HellaSwag dataset
    
    Returns:
        Average attention activations for each layer and head
    """
    # List to store attention activations for each sample
    all_attention_activations = []

    # Process each sample in the dataset
    for tokenized_input in tqdm(tokenized_dataset, desc="Processing Samples"):
        input_ids = tokenized_input['input_ids'].squeeze(0).to(device)
        attention_mask = tokenized_input['attention_mask'].squeeze(0).to(device)

        # Forward pass with output_attentions=True to extract attention weights
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.unsqueeze(0), 
                attention_mask=attention_mask.unsqueeze(0), 
                output_attentions=True
            )
        
        # Extract attention weights
        attention_weights = outputs.attentions
        
        # Convert attention weights to numpy and compute mean activation
        sample_activations = []
        for layer_attn in attention_weights:
            # layer_attn shape: [batch_size, num_heads, seq_len, seq_len]
            layer_mean = layer_attn.mean(dim=(0, 3)).cpu().numpy()
            sample_activations.append(layer_mean)
        
        all_attention_activations.append(sample_activations)

    # Compute average across all samples
    avg_attention_activations = np.mean(all_attention_activations, axis=0)

    return avg_attention_activations

def visualize_attention_activations(avg_attention_activations):
    """
    Visualize average attention activations across layers and heads.
    
    Args:
        avg_attention_activations: NumPy array of average activations
    """
    num_layers, num_heads = avg_attention_activations.shape

    plt.figure(figsize=(15, 8))
    sns.heatmap(
        avg_attention_activations, 
        cmap='YlGnBu', 
        annot=True, 
        fmt='.4f',
        xticklabels=[f'Head {i+1}' for i in range(num_heads)],
        yticklabels=[f'Layer {i+1}' for i in range(num_layers)]
    )
    plt.title('Average Attention Activations Across HellaSwag Samples')
    plt.xlabel('Attention Heads')
    plt.ylabel('Model Layers')
    plt.tight_layout()
    plt.show()

# Prepare dataset
tokenized_dataset = prepare_hellaswag_dataset(tokenizer, num_samples=100)

# Compute and visualize average attention activations
avg_attention_activations = compute_average_attention_activations(model, tokenized_dataset)
visualize_attention_activations(avg_attention_activations)