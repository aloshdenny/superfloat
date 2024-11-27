import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
import gc
import numpy as np

class EnhancedSuperfloat:
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

    def __init__(self, bits: int, adaptive_scaling: bool = True):
        assert 4 <= bits <= 16, "Superfloat bitwidth must be between 4 and 16."
        self.bits = bits
        self.mantissa_bits = bits - 1
        self.max_val = 1 - 2**-self.mantissa_bits  # Precompute max representable value
        self.float_type = self.CASTING_TABLE[bits]  # Get float type based on bitwidth
        self.adaptive_scaling = adaptive_scaling
        
        # Tracking for adaptive scaling
        self.running_max = None
        self.alpha = 0.9  # Exponential moving average decay

    def adaptive_max_val(self, value: torch.Tensor) -> torch.Tensor:
        """Compute adaptive maximum value using exponential moving average."""
        current_max = torch.max(torch.abs(value))
        
        if self.running_max is None:
            self.running_max = current_max
        else:
            self.running_max = self.alpha * self.running_max + (1 - self.alpha) * current_max
        
        return self.running_max

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        """Encodes a tensor of values into the enhanced superfloat format."""
        # Use adaptive or fixed max value
        max_val = self.adaptive_max_val(value) if self.adaptive_scaling else self.max_val
        
        # Clip values to the adaptive or fixed range
        clipped_value = torch.clamp(value, min=-max_val, max=max_val)
        out_of_range = (value.abs() > max_val)
        
        # Compute mantissa and sign
        mantissa = (torch.abs(clipped_value) * (2**self.mantissa_bits - 1) / max_val).floor().to(torch.int32)
        sign = (clipped_value < 0).to(torch.int32)
        
        return (mantissa | (sign << self.mantissa_bits)).to(torch.int32), out_of_range, max_val

    def decode(self, encoded_value: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
        """Decodes a tensor of encoded superfloat values to regular floats."""
        mantissa = encoded_value & ((1 << self.mantissa_bits) - 1)
        sign = (encoded_value >> self.mantissa_bits) & 1
        decoded_value = (mantissa.to(self.float_type) / (2**self.mantissa_bits - 1)) * max_val
        return decoded_value * (2 * sign - 1)

    def tensor_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantizes a tensor to the superfloat format, preserving the tensor's shape."""
        encoded_tensor, out_of_range, max_val = self.encode(tensor)
        decoded_tensor = self.decode(encoded_tensor, max_val)
        return decoded_tensor, out_of_range

class AdvancedQuantizedLlamaModel(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, sf_quantizer: EnhancedSuperfloat):
        super(AdvancedQuantizedLlamaModel, self).__init__()
        self.base_model = base_model
        self.sf_quantizer = sf_quantizer
        self.parameter_importances = {}
        self.apply_gradient_hooks()

    def compute_parameter_importance(self):
        """Compute parameter importance based on gradient magnitudes."""
        self.parameter_importances = {}
        for name, param in self.base_model.named_parameters():
            if param.grad is not None:
                # Compute importance as normalized gradient magnitude
                importance = torch.mean(torch.abs(param.grad)).item()
                self.parameter_importances[name] = importance
        
        # Optional: print top 5 most important parameters
        sorted_importances = sorted(self.parameter_importances.items(), key=lambda x: x[1], reverse=True)
        print("Top 5 Most Important Parameters:")
        for name, importance in sorted_importances[:5]:
            print(f"{name}: {importance}")

    def apply_gradient_hooks(self):
        """Apply advanced gradient hooks with soft masking."""
        def create_hook(param):
            def hook(grad):
                # Quantize the parameter to get out-of-range information
                _, out_of_range, _ = self.sf_quantizer.encode(param)
                
                # Soft gradient masking with exponential decay
                # Parameters far out of range get progressively more dampened gradients
                soft_mask = torch.exp(-torch.abs(out_of_range.to(grad.dtype)) * 5)
                
                return grad * soft_mask
            return hook

        for param in self.base_model.parameters():
            param.register_hook(create_hook(param))

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """Forward pass with adaptive quantization for LLM input."""
        # Quantize model weights before forward pass
        for name, module in self.base_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data, _ = self.sf_quantizer.tensor_quantize(module.weight.data)
        
        # Pass through base model with original arguments
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        
        return outputs

def prepare_model_and_quantization(model_name, sf_bits=11, adaptive_scaling=True):
    """Prepare the model with enhanced Superfloat quantization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Superfloat quantizer
    sf = EnhancedSuperfloat(bits=sf_bits, adaptive_scaling=adaptive_scaling)

    # Load model
    model = LlamaForCausalLM.from_pretrained(
        model_name, 
        cache_dir='./', 
        token='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'
    )
    model = model.to(sf.float_type).to(device)

    # Wrap with advanced quantized model
    quantized_model = AdvancedQuantizedLlamaModel(model, sf)
    
    return quantized_model, sf, device

def prepare_dataset(tokenizer, max_length=1024):
    """Prepare and tokenize dataset."""
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
    """Custom collate function for dataloader."""
    input_ids = torch.stack([torch.tensor(example['input_ids']) for example in batch])
    attention_mask = torch.stack([torch.tensor(example['attention_mask']) for example in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

def train_model(quantized_model, sf, device, num_epochs=3):
    """Training loop with enhanced quantization tracking."""
    # Tokenizer and Dataset preparation
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_name, 
        cache_dir='./', 
        token='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'
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset and dataloader
    tokenized_dataset = prepare_dataset(tokenizer)
    train_dataloader = DataLoader(
        tokenized_dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    # Optimizer and Loss
    optimizer = torch.optim.Adam(
        quantized_model.parameters(), 
        lr=1e-5, 
        eps=1e-4
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training configurations
    accumulation_steps = 32
    best_loss = float('inf')

    quantized_model.to(device)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_iterator = tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader), 
            desc=f"Epoch {epoch + 1}/{num_epochs}"
        )

        for step, batch in epoch_iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = quantized_model(input_ids=input_ids, attention_mask=attention_mask)
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
                # Compute parameter importances
                quantized_model.compute_parameter_importance()

                # Gradient clipping
                torch.nn.utils.clip_grad_value_(
                    quantized_model.parameters(), 
                    clip_value=sf.max_val
                )
                
                optimizer.step()
                optimizer.zero_grad()
                epoch_iterator.set_postfix({"Loss": f"{loss.item() * accumulation_steps:.4f}"})

        epoch_loss /= len(train_dataloader)
        if epoch_loss < best_loss:
            torch.save(
                quantized_model.state_dict(), 
                f"sf{sf.bits}_{epoch+1}_opt"
            )
        print(f"Epoch {epoch + 1} completed with average loss: {epoch_loss:.4f}")

def main():
    # Model and quantization setup
    model_name = "meta-llama/Llama-3.2-1B"
    quantized_model, sf, device = prepare_model_and_quantization(
        model_name, 
        sf_bits=11, 
        adaptive_scaling=True
    )

    # Train the model
    train_model(quantized_model, sf, device)

if __name__ == "__main__":
    main()