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

# SF-Grad Gradient Quantization Function
def superfloat_gradient_quantization(model, gradients, mode='van'):
    """
    Quantize gradients using SuperFloat principles
    
    Args:
        model: Neural network model
        gradients: Computed gradients
        mode: Quantization mode ('van', 'adp', 'pac')
    
    Returns:
        Quantized gradients
    """
    quantized_gradients = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad = param.grad
            
            if mode == 'van':
                # Vanilla: Simple clamping
                quantized_grad = torch.clamp(
                    grad, 
                    min=-1.0, 
                    max=1.0
                )
            
            elif mode == 'adp':
                # Adaptive: Layer-wise dynamic range
                layer_stats = {
                    'mean': grad.mean(),
                    'std': grad.std(),
                    'percentile_99': torch.quantile(grad, 0.99)
                }
                
                quantized_grad = torch.clamp(
                    grad,
                    min=layer_stats['mean'] - 2*layer_stats['std'],
                    max=layer_stats['mean'] + 2*layer_stats['std']
                )
            
            elif mode == 'pac':
                # Precision-Aware: Global and local statistics
                global_mean = torch.mean(grad)
                global_std = torch.std(grad)
                
                quantized_grad = torch.clamp(
                    grad,
                    min=global_mean - 3*global_std,
                    max=global_mean + 3*global_std
                )
            
            quantized_gradients[name] = quantized_grad
    
    return quantized_gradients

def loss_quantization(loss, method='log_clip'):
    """
    Quantize loss value
    
    Args:
        loss: Original loss value
        method: Quantization technique
    
    Returns:
        Quantized loss
    """
    if method == 'log_clip':
        # Logarithmic scaling with clipping
        epsilon = 1e-7
        log_loss = torch.log(loss + epsilon)
        quantized_loss = torch.clamp(
            log_loss, 
            min=-10, 
            max=10
        )
        return torch.exp(quantized_loss)
    
    elif method == 'percentile_clip':
        # Percentile-based clipping
        loss_percentile = torch.quantile(loss, 0.95)
        return torch.min(loss, loss_percentile)

class QuantizedLlamaModel(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, sf_quantizer: Superfloat, grad_mode='van'):
        super(QuantizedLlamaModel, self).__init__()
        self.base_model = base_model
        self.sf_quantizer = sf_quantizer
        self.grad_mode = grad_mode
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

# Initialization and Hyperparameters
sf = Superfloat(11)  # 11-bit Superfloat configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "meta-llama/Llama-3.2-1B"
model = LlamaForCausalLM.from_pretrained(model_name, cache_dir='./', token='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')
model = model.to(sf.float_type).to(device)

# Tokenizer Setup
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, cache_dir='./', token='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')
tokenizer.pad_token = tokenizer.eos_token

# Quantization of Model Weights
def quantize_model(model, sf_type):
    for name, param in model.named_parameters():
        quantized_param, _ = sf_type.tensor_quantize(param)
        param.data = quantized_param.data
    return model

# Checkpoint Loading Function
import os
import re

def load_checkpoint(model, sf_bits, suffix="opt", device="cuda"):
    checkpoint_pattern = re.compile(f"sf{sf_bits}_.*_epoch(\\d+)_.*{suffix}$")

    checkpoint_files = [
        f for f in os.listdir(".") if checkpoint_pattern.match(f)
    ]

    if not checkpoint_files:
        print(f"No checkpoints found for sf{sf_bits} with suffix '{suffix}'.")
        return quantize_model(model, sf), 0

    epochs_and_files = [
        (int(checkpoint_pattern.match(f).group(1)), f) for f in checkpoint_files
    ]
    latest_epoch, latest_checkpoint = max(epochs_and_files, key=lambda x: x[0])

    print(f"Loading checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    return model, latest_epoch

# Dataset Preparation
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

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(example['input_ids']) for example in batch])
    attention_mask = torch.stack([torch.tensor(example['attention_mask']) for example in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

# Main Training Script
def train_model(model, sf, tokenizer, device, grad_mode='van', loss_mode='log_clip'):
    # Prepare Quantized Model
    quantized_model = QuantizedLlamaModel(model, sf, grad_mode)
    quantized_model.to(device)

    # Checkpoint Loading (Optional)
    quantized_model, last_epoch = load_checkpoint(quantized_model, sf.bits, suffix="opt", device=device)
    print(f"Resuming training from epoch {last_epoch + 1}.")

    # Prepare Dataset and Dataloader
    tokenized_dataset = prepare_dataset(tokenizer)
    train_dataloader = DataLoader(tokenized_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(quantized_model.parameters(), lr=1e-5, eps=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training Configuration
    num_epochs = 3
    accumulation_steps = 32
    best_loss = float('inf')

    # Training Loop
    for epoch in range(last_epoch, last_epoch + num_epochs):
        epoch_loss = 0.0
        epoch_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}")

        for step, batch in epoch_iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = quantized_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            target = input_ids[:, 1:].contiguous()
            logits = logits[:, :-1].contiguous()

            # Calculate and Quantize Loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1)) / accumulation_steps
            quantized_loss = loss_quantization(loss, method=loss_mode)

            # Backward pass with gradient quantization
            quantized_loss.backward()

            # Accumulate loss for reporting
            epoch_loss += quantized_loss.item() * accumulation_steps

            if (step + 1) % accumulation_steps == 0:
                # Apply SF-Grad Gradient Quantization
                quantized_grads = superfloat_gradient_quantization(quantized_model, quantized_model.parameters(), mode=grad_mode)
                
                # Apply quantized gradients manually
                for name, param in quantized_model.named_parameters():
                    if param.requires_grad and name in quantized_grads:
                        param.grad = quantized_grads[name]

                # Gradient Clipping
                torch.nn.utils.clip_grad_value_(quantized_model.parameters(), clip_value=sf.max_val)
                
                optimizer.step()
                optimizer.zero_grad()
                epoch_iterator.set_postfix({"Loss": f"{quantized_loss.item() * accumulation_steps:.4f}"})

        # Loss and Checkpoint Handling
        epoch_loss /= len(train_dataloader)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_path = f"sf{sf.bits}_{epoch+1}_opt"
            torch.save(quantized_model.state_dict(), checkpoint_path)
            print(f"New best model saved to {checkpoint_path}")

        print(f"Epoch {epoch + 1} completed with average loss: {epoch_loss:.4f}")

# Example Usage
if __name__ == "__main__":
    # Different Gradient Quantization Modes
    grad_modes = ['van', 'adp', 'pac']
    loss_modes = ['log_clip', 'percentile_clip']

    for grad_mode in grad_modes:
        for loss_mode in loss_modes:
            print(f"\nTraining with Gradient Mode: {grad_mode}, Loss Mode: {loss_mode}")
            # Reset model for each configuration
            reset_model = LlamaForCausalLM.from_pretrained(model_name, cache_dir='./', token='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')
            train_model(reset_model, sf, tokenizer, device, grad_mode, loss_mode)