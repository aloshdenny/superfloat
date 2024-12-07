import os
import re
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np

class QuantizationMetrics:
    def __init__(self):
        self.out_of_range_params = []
        self.layer_bitwidths = []
    
    def track(self, model, sf_quantizer):
        for name, param in model.named_parameters():
            _, out_of_range = sf_quantizer.tensor_quantize(param)
            self.out_of_range_params.append(out_of_range.float().mean())

class AdaptiveQuantizationMask(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.quantization_mask = nn.Parameter(torch.ones(layer_size, requires_grad=True))
        
    def forward(self, tensor):
        return tensor * self.quantization_mask

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
        self.max_val = 1 - 2**-self.mantissa_bits
        self.float_type = self.CASTING_TABLE[bits]

    def adaptive_bitwidth(self, layer_weights):
        """Dynamically adjust bit allocation based on layer importance"""
        weight_variance = torch.var(layer_weights)
        return min(16, max(4, int(8 + weight_variance * 4)))

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

def quantization_loss(model, lambda_sparse=0.001):
    """Add sparsity-inducing loss to the model"""
    sparse_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
    return sparse_loss * lambda_sparse

class QuantizedLlamaModel(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, sf_quantizer: Superfloat):
        super(QuantizedLlamaModel, self).__init__()
        self.base_model = base_model
        self.sf_quantizer = sf_quantizer
        self.quantization_metrics = QuantizationMetrics()
        self.adaptive_masks = nn.ModuleDict()
        self.apply_gradient_hooks()
        self.prepare_adaptive_masks()

    def prepare_adaptive_masks(self):
        """Create adaptive quantization masks for each layer"""
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                self.adaptive_masks[name.replace('.', '_')] = AdaptiveQuantizationMask(param.size())

    def apply_gradient_hooks(self):
        """Apply quantization and gradient hooks to model parameters"""
        for param in self.base_model.parameters():
            def hook(grad, param=param):
                # Track out-of-range parameters
                _, out_of_range = self.sf_quantizer.tensor_quantize(param)
                self.quantization_metrics.track(self, self.sf_quantizer)

                # Adaptive gradient masking
                grad = grad * out_of_range.to(grad.dtype)
                return grad
            param.register_hook(hook)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """Forward pass with quantization at each layer"""
        # Quantize input tensors
        input_ids, _ = self.sf_quantizer.tensor_quantize(input_ids)
        attention_mask, _ = self.sf_quantizer.tensor_quantize(attention_mask)

        # Apply quantization and adaptive masks to model weights
        for name, layer in self.base_model.named_children():
            if isinstance(layer, torch.nn.Linear):
                mask_name = name.replace('.', '_')
                layer.weight.data = layer.weight.data * self.adaptive_masks[mask_name](layer.weight.data)
                layer.weight.data, _ = self.sf_quantizer.tensor_quantize(layer.weight.data)

        # Forward through the base model
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

def prepare_dataset(tokenizer, max_length=1024):
    """Prepare and tokenize dataset"""
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

def collate_fn(batch):
    """Custom collate function for dataloader"""
    input_ids = torch.stack([torch.tensor(example['input_ids']) for example in batch])
    attention_mask = torch.stack([torch.tensor(example['attention_mask']) for example in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

def load_checkpoint(model, sf_bits, suffix="opt", device="cuda"):
    """Load the latest checkpoint based on Superfloat bitwidth"""
    checkpoint_pattern = re.compile(f"sf{sf_bits}_.*_epoch(\\d+)_.*{suffix}$")
    checkpoint_files = [f for f in os.listdir(".") if checkpoint_pattern.match(f)]

    if not checkpoint_files:
        print(f"No checkpoints found for sf{sf_bits} with suffix '{suffix}'.")
        return model, 0

    epochs_and_files = [(int(checkpoint_pattern.match(f).group(1)), f) for f in checkpoint_files]
    latest_epoch, latest_checkpoint = max(epochs_and_files, key=lambda x: x[0])

    print(f"Loading checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    return model, latest_epoch

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Superfloat Configuration
    sf = Superfloat(11)  # Adaptive bitwidth can be implemented here

    # Model and Tokenizer
    model_name = "meta-llama/Llama-3.2-1B"
    model = LlamaForCausalLM.from_pretrained(model_name, cache_dir='./', token='your_huggingface_token')
    model = model.to(sf.float_type).to(device)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, cache_dir='./', token='your_huggingface_token')
    tokenizer.pad_token = tokenizer.eos_token

    # Quantized Model Wrapper
    quantized_model = QuantizedLlamaModel(model, sf)
    quantized_model, last_epoch = load_checkpoint(quantized_model, sf.bits, suffix="opt", device=device)

    # Prepare Dataset
    tokenized_dataset = prepare_dataset(tokenizer)
    train_dataloader = DataLoader(tokenized_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(quantized_model.parameters(), lr=1e-4, eps=1e-8, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-6)
    
    # Mixed Precision
    scaler = GradScaler()

    # Training Loop
    num_epochs = 3
    accumulation_steps = 32
    best_loss = float('inf')

    quantized_model.to(device)
    quantized_model.train()

    for epoch in range(last_epoch + 1, last_epoch + num_epochs + 1):
        epoch_loss = 0.0
        epoch_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                               desc=f"Epoch {epoch}/{last_epoch + num_epochs}")

        for step, batch in epoch_iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with autocast(enabled=True):
                outputs = quantized_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                target = input_ids[:, 1:].contiguous()
                logits = logits[:, :-1].contiguous()

                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
                
                # Add quantization loss
                loss += quantization_loss(quantized_model)

                scaled_loss = loss / accumulation_steps

            scaler.scale(scaled_loss).backward()

            if (step + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(quantized_model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_iterator.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Learning rate scheduling
        scheduler.step()

        # Average epoch loss
        epoch_loss /= len(train_dataloader)
        print(f"Epoch {epoch} completed with average loss: {epoch_loss:.4f}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(quantized_model.state_dict(), f"sf{sf.bits}_{epoch}_best_opt")

        # Cleanup
        del input_ids, attention_mask, outputs
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()