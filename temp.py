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

class QuantizedLlamaModel(nn.Module):
    def __init__(self, base_model: nn.Module, sf_quantizer: Superfloat):
        super(QuantizedLlamaModel, self).__init__()
        self.base_model = base_model
        self.sf_quantizer = sf_quantizer
        self.dtype = self.sf_quantizer.float_type
        self._quantize_and_convert_parameters()
        self.apply_gradient_hooks()

    def _quantize_and_convert_parameters(self):
        """Quantize all model parameters and ensure consistent dtype"""
        for name, param in self.base_model.named_parameters():
            if param.dtype != self.dtype:
                param.data = param.data.to(self.dtype)
            quantized_param, _ = self.sf_quantizer.tensor_quantize(param.data)
            param.data.copy_(quantized_param)

    def apply_gradient_hooks(self):
        """Apply gradient hooks to all parameters for quantization during training"""
        for param in self.base_model.parameters():
            if param.requires_grad:
                def hook(grad, param=param):
                    if grad.dtype != self.dtype:
                        grad = grad.to(self.dtype)
                    quantized_grad, _ = self.sf_quantizer.tensor_quantize(grad)
                    _, out_of_range = self.sf_quantizer.tensor_quantize(param)
                    masked_grad = quantized_grad * out_of_range.to(grad.dtype)
                    return masked_grad
                param.register_hook(hook)

    def _quantize_activation(self, x):
        """Quantize activation tensors with dtype handling"""
        if not x.is_floating_point():  # Skip quantization for non-float tensors
            return x
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        quantized_x, _ = self.sf_quantizer.tensor_quantize(x)
        return quantized_x

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Keep input_ids as Long type for embedding layer
        # Only convert attention_mask to float dtype if it exists
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.dtype)
        
        # Handle input quantization
        def quantize_module_fn(module, inputs, outputs):
            if isinstance(outputs, torch.Tensor):
                if outputs.is_floating_point():  # Only quantize float tensors
                    return self._quantize_activation(outputs)
                return outputs
            elif isinstance(outputs, tuple):
                return tuple(self._quantize_activation(o) if isinstance(o, torch.Tensor) and o.is_floating_point() else o 
                           for o in outputs)
            return outputs

        # Register forward hooks for all modules
        hooks = []
        for name, module in self.base_model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                hook = module.register_forward_hook(quantize_module_fn)
                hooks.append(hook)

            # Ensure all module parameters have correct dtype
            for param_name, param in module.named_parameters(recurse=False):
                if param.dtype != self.dtype and param.is_floating_point():
                    param.data = param.data.to(self.dtype)

        try:
            # Forward pass through the model
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        finally:
            # Remove the hooks
            for hook in hooks:
                hook.remove()

        return outputs

    def generate(self, *args, **kwargs):
        """Handle generation while maintaining correct dtypes"""
        # Convert only float tensors to the correct dtype
        converted_args = [
            arg.to(self.dtype) if isinstance(arg, torch.Tensor) and arg.is_floating_point()
            else arg for arg in args
        ]
        converted_kwargs = {
            k: v.to(self.dtype) if isinstance(v, torch.Tensor) and v.is_floating_point()
            else v for k, v in kwargs.items()
        }
        return self.base_model.generate(*converted_args, **converted_kwargs)

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

import os
import re

def load_checkpoint(model, sf_bits, suffix="opt", device="cuda"):
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
        return QuantizedLlamaModel(model, sf), 0

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

# Usage
quantized, last_epoch = load_checkpoint(model, sf.bits, suffix="opt", device=device)
print(f"Resuming training from epoch {last_epoch + 1}.")

del model
torch.cuda.empty_cache()
gc.collect()

# Prepare Dataset
def prepare_dataset(tokenizer, max_length=1024):
    dataset = Dataset.from_parquet('train.parquet')
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
optimizer = torch.optim.Adam(quantized.parameters(), lr=1e-5, eps=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Training Loop
num_epochs = 3
accumulation_steps = 32  # Number of steps to accumulate gradients
best_loss = float('inf')

# Training script changes
quantized.to(device)

optimizer = torch.optim.Adam(quantized.parameters(), lr=1e-5, eps=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 3
accumulation_steps = 32  # Number of steps to accumulate gradients
best_loss = float('inf')

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

        # Backward pass
        loss.backward()

        # Accumulate loss for reporting
        epoch_loss += loss.item() * accumulation_steps

        if (step + 1) % accumulation_steps == 0:
            # Apply gradient updates
            optimizer.step()
            optimizer.zero_grad()

            # Clamp only out-of-range parameters
            for param in quantized.parameters():
                _, out_of_range = sf.tensor_quantize(param.data)
                param.data[out_of_range] = torch.clamp(param.data[out_of_range], min=-sf.max_val, max=sf.max_val)

            epoch_iterator.set_postfix({"Loss": f"{loss.item() * accumulation_steps:.4f}"})

    epoch_loss /= len(train_dataloader)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(quantized.state_dict(), f"sf{sf.bits}_{epoch + 1}_opt")
    print(f"Epoch {epoch + 1} completed with average loss: {epoch_loss:.4f}")