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
        8: torch.float16,
        7: torch.float16,
        6: torch.float16,
        5: torch.float16,
        4: torch.float16,
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

sf = Superfloat(11)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "meta-llama/Llama-3.2-1B"
model = LlamaForCausalLM.from_pretrained(model_name, cache_dir='./', token='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')
model = model.to(sf.float_type).to(device)

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, cache_dir='./', token='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')
tokenizer.pad_token = tokenizer.eos_token

# Quantize Model Weights Selectively
def quantize_model(model, sf_type):
    for name, param in model.named_parameters():
        quantized_param, _ = sf_type.tensor_quantize(param)
        param.data = quantized_param.data
    return model

quantized = quantize_model(model, sf)

import os
import re

def load_checkpoint(quantized_model, sf_bits, suffix="opt", device="cuda"):
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
        return quantized_model, 0

    # Extract epoch numbers and sort by latest epoch
    epochs_and_files = [
        (int(checkpoint_pattern.match(f).group(1)), f) for f in checkpoint_files
    ]
    latest_epoch, latest_checkpoint = max(epochs_and_files, key=lambda x: x[0])

    # Load the latest checkpoint
    print(f"Loading checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    quantized_model.load_state_dict(checkpoint)
    quantized_model.to(device)

    return quantized_model, latest_epoch

# Usage
quantized, last_epoch = load_checkpoint(quantized, sf.bits, suffix="opt", device=device)
print(f"Resuming training from epoch {last_epoch + 1}.")

del model
torch.cuda.empty_cache()
gc.collect()

# Prepare Dataset
def prepare_dataset(tokenizer, max_length=1024):
    dataset = Dataset.from_parquet('train-00000-of-01650-f70471ee3deb09c0.parquet')
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
tokenized_dataset = prepare_dataset(tokenizer)
train_dataloader = DataLoader(tokenized_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Optimizer and Loss
optimizer = torch.optim.Adam(quantized.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training Loop
num_epochs = 3
accumulation_steps = 32  # Number of steps to accumulate gradients
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
            optimizer.step()
            optimizer.zero_grad()
            epoch_iterator.set_postfix({"Loss": f"{loss.item() * accumulation_steps:.4f}"})

    epoch_loss /= len(train_dataloader)
    torch.save(quantized.state_dict(), f"sf{sf.bits}_pile_epoch{epoch+1}_opt")
    print(f"Epoch {epoch + 1} completed with average loss: {epoch_loss:.4f}")