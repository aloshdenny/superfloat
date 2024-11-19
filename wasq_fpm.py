import torch

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
        """Quantizes a tensor to the superfloat format, preserving the tensor's shape."""
        # Apply element-wise encoding to the entire tensor and then decode back
        print(f"Params in layer: {len(tensor)}")
        encoded_tensor = self.encode(tensor)
        print("Encoding complete")
        decoded_tensor = self.decode(encoded_tensor)
        print("Decoding complete")
        return decoded_tensor

sf = Superfloat(8)  # Create an SF8 superfloat type

class QuantizedLlamaModel(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, sf_quantizer: Superfloat):
        super(QuantizedLlamaModel, self).__init__()
        self.base_model = base_model
        self.sf_quantizer = sf_quantizer
        self.apply_gradient_hooks()

    def apply_gradient_hooks(self):
        # Register a hook to quantize gradients after backward pass
        for param in self.base_model.parameters():
            param.register_hook(lambda grad: self.sf_quantizer.tensor_quantize(grad))

    def forward(self, x):
        # Quantize activations and parameters during forward pass
        x = self.sf_quantizer.tensor_quantize(x)
        for layer in self.base_model.children():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data = self.sf_quantizer.tensor_quantize(layer.weight.data)
            x = self.sf_quantizer.tensor_quantize(layer(x))
        return x

from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
model = LlamaForCausalLM.from_pretrained(model_name, cache_dir='./', token='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')
model = model.to(sf.float_type).to(device)

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, cache_dir='./', token='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')

tokenizer.pad_token = tokenizer.eos_token

def quantize_model(model, sf_type):
    for name, param in model.named_parameters():
        print(name, len(param))
        quantized_param = sf_type.tensor_quantize(param)
        param.data = quantized_param.data
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

quantized = quantize_model(model, sf)

import os
import re

def load_checkpoint(quantized_model, sf_bits, suffix="fpm", device="cuda"):
    """
    Load the latest checkpoint based on the provided Superfloat bitwidth and filename suffix.
    
    Args:
        quantized_model: The model to load the checkpoint into.
        sf_bits: Bitwidth of the Superfloat format (e.g., 11).
        suffix: The suffix of the filename (default: 'fpm').
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
quantized, last_epoch = load_checkpoint(quantized, sf.bits, suffix="fpm", device=device)
print(f"Resuming training from epoch {last_epoch + 1}.")

del model
torch.cuda.empty_cache()
import gc
gc.collect()

from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset

# Prepare Dataset
def prepare_dataset(tokenizer, max_length=512):
    """Prepare the dataset with proper tensor formatting."""
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = Dataset.from_parquet('train-00000-of-01650-f70471ee3deb09c0.parquet')
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return outputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset

# Custom collate function
def collate_fn(batch):
    """Custom collate function to properly format tensors."""
    input_ids = torch.stack([torch.tensor(example['input_ids']) for example in batch])
    attention_mask = torch.stack([torch.tensor(example['attention_mask']) for example in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

# Prepare tokenized dataset and dataloader
tokenized_dataset = prepare_dataset(tokenizer)
train_dataloader = DataLoader(tokenized_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

from tqdm import tqdm

# Optimizer and Loss
optimizer = torch.optim.Adam(quantized.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 3
accumulation_steps = 32  # Number of steps to accumulate gradients
best_loss = float('inf')

quantized.to(device)

# Training Loop with Autoregressive Target, Gradient Accumulation, and Progress Tracking
for epoch in range(num_epochs):
    epoch_loss = 0.0  # Track total loss for each epoch

    # Initialize tqdm for tracking epoch progress
    epoch_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")

    for step, batch in epoch_iterator:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass
        outputs = quantized(input_ids=input_ids, attention_mask=attention_mask)

        # Access the logits for token predictions
        logits = outputs.logits  # Retrieve logits tensor from ModelOutput

        # Shift input_ids by one for autoregressive target
        target = input_ids[:, 1:].contiguous()  # Target is the input shifted by one token
        logits = logits[:, :-1].contiguous()    # Align logits with target length

        # Calculate loss
        loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))

        # Divide loss by accumulation steps
        loss = loss / accumulation_steps

        # Backward pass with gradient quantization
        loss.backward()  # Gradient quantization occurs via hook

        # Accumulate the loss for reporting
        epoch_loss += loss.item() * accumulation_steps

        # Perform optimizer step after accumulating gradients
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()  # Clear gradients for next accumulation

            # Update tqdm progress bar with current step loss
            epoch_iterator.set_postfix({"Loss": f"{loss.item() * accumulation_steps:.4f}"})

    # Average epoch loss
    epoch_loss /= len(train_dataloader)
    if epoch_loss < best_loss:
        torch.save(quantized.state_dict(), f"sf{sf.bits}_pile_epoch{epoch+1}")
    print(f"Epoch {epoch + 1} completed with average loss: {epoch_loss:.4f}")