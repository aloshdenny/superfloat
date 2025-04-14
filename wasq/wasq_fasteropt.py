import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
import gc

max_length = 512
bit = 8

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
        out_of_range = (value.abs() > self.max_val)
        mantissa = (torch.abs(clipped_value) * (2**self.mantissa_bits - 1) / self.max_val).floor().to(torch.int32)
        sign = (clipped_value < 0).to(torch.int32)
        return (mantissa | (sign << self.mantissa_bits)).to(torch.int32), out_of_range

    def decode(self, encoded_value: torch.Tensor) -> torch.Tensor:
        """Decodes a tensor of encoded superfloat values to regular floats."""
        mantissa = encoded_value & ((1 << self.mantissa_bits) - 1)
        sign = (encoded_value >> self.mantissa_bits) & 1
        decoded_value = (mantissa.to(self.float_type) / (2**self.mantissa_bits - 1)) * self.max_val
        return decoded_value * (2 * sign - 1)

    def tensor_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantizes a tensor to the superfloat format, preserving the tensor's shape."""
        encoded_tensor, out_of_range = self.encode(tensor)   
        decoded_tensor = self.decode(encoded_tensor)
        return decoded_tensor, out_of_range

sf = Superfloat(bit)

class QuantizedLlamaModel(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, sf_quantizer: Superfloat):
        super(QuantizedLlamaModel, self).__init__()
        self.base_model = base_model
        self.sf_quantizer = sf_quantizer
        self.apply_gradient_hooks()

    def apply_gradient_hooks(self):
        for param in self.base_model.parameters():
            def hook(grad, param=param):
                _, out_of_range = self.sf_quantizer.tensor_quantize(param)
                grad = grad * out_of_range.to(grad.dtype)  # Mask to allow gradients only on out-of-range params
                return grad
            param.register_hook(hook)

    def forward(self, x):
        x, _ = self.sf_quantizer.tensor_quantize(x)
        for layer in self.base_model.children():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data, _ = self.sf_quantizer.tensor_quantize(layer.weight.data)
            x = layer(x)
            x, _ = self.sf_quantizer.tensor_quantize(x)
        return x

# Initialize model and tokenizer
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model_name = "Qwen/Qwen2-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='./', token='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')
model = model.to(sf.float_type).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./', token='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')
tokenizer.pad_token = tokenizer.eos_token

# Quantize Model Weights Selectively
def quantize_model(model, sf_type):
    for name, param in model.named_parameters():
        quantized_param, _ = sf_type.tensor_quantize(param)
        param.data = quantized_param.data
    return model

import os
import re

def load_checkpoint(model, sf_bits, suffix="opt", device=device):
    """
    Load the latest checkpoint based on the provided Superfloat bitwidth and filename suffix.

    Args:
        quantized_model: The model to load the checkpoint into.
        sf_bits: Bitwidth of the Superfloat format (e.g., 11).
        suffix: The suffix of the filename (default: 'opt').
        device: Device to load the model onto ('cuda' or 'cpu').

    Returns:
        The quantized model with loaded weights and the epoch number.
    """
    # Define the filename pattern to search for
    checkpoint_pattern = re.compile(f"sf{sf_bits}_.*_epoch(\\d+)_.*{suffix}$")

    # Find all matching checkpoint files
    checkpoint_files = [
        f for f in os.listdir(".") if checkpoint_pattern.match(f)
    ]

    if not checkpoint_files:
        print(f"No checkpoints found for sf{sf_bits} with suffix '{suffix}'.")
        return quantize_model(model, sf), 0

    # Extract epoch numbers and sort by latest epoch
    epochs_and_files = [
        (int(checkpoint_pattern.match(f).group(1)), f) for f in checkpoint_files
    ]
    latest_epoch, latest_checkpoint = max(epochs_and_files, key=lambda x: x[0])

    # Load the latest checkpoint
    print(f"Loading checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    return model, latest_epoch

# Pre-training parameter check to ensure they are within range
def check_parameters_in_range(model, sf):
    out_of_range_params = []
    for name, param in model.named_parameters():
        if not torch.all(torch.abs(param.data) <= sf.max_val):
            out_of_range_params.append(name)
    if out_of_range_params:
        print(f"Warning: The following parameters are out of range:")
        for param_name in out_of_range_params:
            print(f"- {param_name}")
    else:
        print("All parameters are within the valid range.")

# Usage
quantized, last_epoch = load_checkpoint(model, sf.bits, suffix="opt", device=device)
print(f"Resuming training from epoch {last_epoch + 1}.")

# Check if model parameters are within range before training
check_parameters_in_range(quantized, sf)

del model
if device=="cuda":
    torch.cuda.empty_cache()
elif device=="mps":
    torch.mps.empty_cache()
gc.collect()

# Prepare Dataset
def prepare_dataset(tokenizer, max_length=1):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # dataset = Dataset.from_parquet('train.parquet')
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset

# Custom collate function
def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(example['input_ids']) for example in batch])
    attention_mask = torch.stack([torch.tensor(example['attention_mask']) for example in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

# Prepare tokenized dataset and dataloader
tokenized_dataset = prepare_dataset(tokenizer, max_length=max_length)
train_dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Optimizer and Loss
optimizer = torch.optim.Adam(quantized.parameters(), lr=1e-5, eps=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Training Loop
num_epochs = 3
accumulation_steps = 8  # Number of steps to accumulate gradients
best_loss = float('inf')

quantized.to(device)

for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")

    for step, batch in epoch_iterator:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass
        outputs = quantized(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        target = input_ids[:, 1:].contiguous()
        logits = logits[:, :-1].contiguous()

        # Calculate loss
        loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1)) / accumulation_steps

        # Backward pass with gradient quantization
        loss.backward()

        # Accumulate loss for reporting
        epoch_loss += loss.item() * accumulation_steps

        if (step + 1) % accumulation_steps == 0:
            # Clamp activations and model parameters within the Superfloat range
            for name, param in quantized.named_parameters():
                param.data = torch.clamp(param.data, min=-sf.max_val, max=sf.max_val)

            # Check activations range
            for name, param in quantized.named_parameters():
                if not torch.all(torch.abs(param.data) <= sf.max_val):
                    print(f"Warning: {name} activation is out of range after clamping!")

            torch.nn.utils.clip_grad_value_(quantized.parameters(), clip_value=sf.max_val)
            optimizer.step()
            optimizer.zero_grad()
            epoch_iterator.set_postfix({"Loss": f"{loss.item() * accumulation_steps:.4f}"})

    epoch_loss /= len(train_dataloader)
    if epoch_loss < best_loss:
        torch.save(quantized.state_dict(), f"sf{sf.bits}_{epoch+1}_opt")
    print(f"Epoch {epoch + 1} completed with average loss: {epoch_loss:.4f}")