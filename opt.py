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
        self.max_val = 2**(self.mantissa_bits - 1) - 1  # Adjusted max value calculation
        self.float_type = self.CASTING_TABLE[bits]

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        """Encode tensor to Superfloat representation."""
        # Normalize value to the range of representable values
        scaled = torch.clamp(value, min=-self.max_val, max=self.max_val)
        
        # Convert to fixed-point representation
        scaled_int = torch.round(scaled).to(torch.int32)
        
        # Encode sign bit
        sign_bit = (scaled_int < 0).to(torch.int32)
        abs_val = torch.abs(scaled_int)
        
        # Combine sign and magnitude
        encoded = (abs_val | (sign_bit << (self.mantissa_bits)))
        
        return encoded

    def decode(self, encoded_value: torch.Tensor) -> torch.Tensor:
        """Decode Superfloat representation back to floating-point."""
        # Extract magnitude and sign
        magnitude = encoded_value & ((1 << self.mantissa_bits) - 1)
        sign_bit = (encoded_value >> self.mantissa_bits) & 1
        
        # Convert back to signed integer
        signed_val = magnitude.to(self.float_type)
        signed_val = signed_val * (2 * sign_bit - 1)
        
        return signed_val

    def tensor_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor using Superfloat encoding and decoding."""
        # Encode to Superfloat 
        encoded = self.encode(tensor)
        
        # Decode back to floating point
        decoded = self.decode(encoded)
        
        # Track out-of-range values
        out_of_range = (tensor.abs() > self.max_val)
        
        return decoded, out_of_range

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM

class QuantizedLlamaModel(nn.Module):
    def __init__(self, base_model: LlamaForCausalLM, sf_quantizer):
        super().__init__()
        self.base_model = base_model
        self.sf_quantizer = sf_quantizer
        
        # Quantize model parameters during initialization
        self.quantize_model_parameters()

    def quantize_model_parameters(self):
        """Quantize all model parameters during initialization."""
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                quantized_param, _ = self.sf_quantizer.tensor_quantize(param.data)
                param.data.copy_(quantized_param)

    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        past_key_values=None, 
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        **kwargs
    ):
        """
        Forward method that matches the original LlamaForCausalLM forward signature
        and applies Superfloat quantization.
        """
        # Ensure input_ids is long tensor for embedding layer
        if input_ids is not None:
            # Convert to float for quantization, then back to long
            input_ids_float, _ = self.sf_quantizer.tensor_quantize(input_ids.float())
            input_ids = input_ids_float.long()

        # If inputs_embeds is provided, quantize it
        if inputs_embeds is not None:
            inputs_embeds, _ = self.sf_quantizer.tensor_quantize(inputs_embeds)

        # Perform forward pass on the base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
            **kwargs
        )

        # Quantize logits if they exist
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            outputs.logits, _ = self.sf_quantizer.tensor_quantize(outputs.logits)

        return outputs

    def __getattr__(self, name):
        """
        Delegate attribute access to the base model to maintain 
        compatibility with original model methods.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

# Main script remains largely the same, but use the improved QuantizedLlamaModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Superfloat configuration
sf = Superfloat(bits=11)  # 11-bit Superfloat 

# Load model
model_name = "Qwen/Qwen2-0.5B"
model = LlamaForCausalLM.from_pretrained(model_name, cache_dir='./', token='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')
model = model.to(sf.float_type).to(device)

# Wrap model with Quantized Wrapper
quantized_model = QuantizedLlamaModel(model, sf)

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
import gc
import os
import re

def load_checkpoint(quantized_model, sf_bits, suffix="opt", device="cuda"):
    """
    Load the latest checkpoint based on the provided Superfloat bitwidth and filename suffix.

    Args:
        quantized_model: The quantized model to load the checkpoint into.
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
    
    # Carefully load state dict
    try:
        quantized_model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return quantized_model, 0

    quantized_model.to(device)

    return quantized_model, latest_epoch

def prepare_dataset(tokenizer, max_length=1024):
    """
    Prepare and tokenize dataset from Parquet file.

    Args:
        tokenizer: Tokenizer to use for encoding.
        max_length: Maximum sequence length.

    Returns:
        Tokenized dataset.
    """
    try:
        dataset = Dataset.from_parquet('train.parquet')
    except FileNotFoundError:
        print("Error: train.parquet file not found. Please check the file path.")
        return None

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    return tokenized_dataset

def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle tokenized data.

    Args:
        batch: Batch of tokenized examples.

    Returns:
        Collated batch with input_ids and attention_mask.
    """
    input_ids = torch.stack([torch.tensor(example['input_ids']) for example in batch])
    attention_mask = torch.stack([torch.tensor(example['attention_mask']) for example in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

def main():
    # Device and hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Superfloat configuration
    sf = Superfloat(bits=11)  # Configurable bit-width

    # Model and Tokenizer Setup
    model_name = "Qwen/Qwen2-0.5B"
    
    try:
        model = LlamaForCausalLM.from_pretrained(
            model_name, 
            cache_dir='./', 
            token='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'
        )
        model = model.to(sf.float_type).to(device)

        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_name, 
            cache_dir='./', 
            token='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'
        )
        tokenizer.pad_token = tokenizer.eos_token

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    # Wrap model with Quantized Wrapper
    quantized_model = QuantizedLlamaModel(model, sf)

    # Load checkpoint (if exists)
    quantized_model, last_epoch = load_checkpoint(quantized_model, sf.bits, suffix="opt", device=device)
    print(f"Resuming training from epoch {last_epoch + 1}.")

    # Clear original model to free up memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Prepare Dataset
    tokenized_dataset = prepare_dataset(tokenizer)
    if tokenized_dataset is None:
        return

    # Create DataLoader
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

    # Training Configuration
    num_epochs = 3
    accumulation_steps = 32
    best_loss = float('inf')

    # Move model to device
    quantized_model.to(device)

    # Training Loop
    for epoch in range(last_epoch, num_epochs):
        quantized_model.train()
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
            
            # Prepare target (next token prediction)
            target = input_ids[:, 1:].contiguous()
            logits = logits[:, :-1].contiguous()

            # Calculate loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1)) / accumulation_steps

            # Backward pass with gradient quantization
            loss.backward()

            # Accumulate loss for reporting
            epoch_loss += loss.item() * accumulation_steps

            if (step + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_value_(
                    quantized_model.parameters(), 
                    clip_value=sf.max_val
                )
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Update progress bar
                epoch_iterator.set_postfix({"Loss": f"{loss.item() * accumulation_steps:.4f}"})

        # Calculate average epoch loss
        epoch_loss /= len(train_dataloader)
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_path = f"sf{sf.bits}_epoch{epoch+1}_opt"
            torch.save(quantized_model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

        print(f"Epoch {epoch + 1} completed with average loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    main()