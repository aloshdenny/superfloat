import torch

class Superfloat:
    def __init__(self, bits: int):
        assert 4 <= bits <= 16, "Superfloat bitwidth must be between 4 and 16."
        self.bits = bits
        self.mantissa_bits = bits - 1
        self.max_val = 1 - 2**-self.mantissa_bits  # Precompute max representable value

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        """Encodes a tensor of values into the superfloat format with optimized operations."""
        # Clip tensor values to the valid range for SFx
        clipped_value = torch.clamp(value, min=-self.max_val, max=self.max_val)

        # Calculate mantissa representation element-wise
        mantissa = (torch.abs(clipped_value) * (2**self.mantissa_bits - 1) / self.max_val).floor().to(torch.int32)

        # Create the superfloat representation (1 bit for sign and mantissa bits)
        sign = (clipped_value < 0).to(torch.int32)
        return (mantissa | (sign << self.mantissa_bits)).to(torch.int32)

    def decode(self, encoded_value: torch.Tensor) -> torch.Tensor:
        """Decodes a tensor of encoded superfloat values to regular floats."""
        # Extract mantissa and sign from the encoded superfloat
        mantissa = encoded_value & ((1 << self.mantissa_bits) - 1)
        sign = (encoded_value >> self.mantissa_bits) & 1

        # Calculate the decoded float using the mantissa and max_val
        decoded_value = (mantissa.to(torch.bfloat16) / (2**self.mantissa_bits - 1)) * self.max_val
        return decoded_value * (2 * sign - 1)  # Apply the sign

    def tensor_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantizes a tensor to the superfloat format, preserving the tensor's shape."""
        # Apply element-wise encoding to the entire tensor and then decode back
        print(f"Params in layer: {len(tensor)}")
        encoded_tensor = self.encode(tensor)
        print("Encoding complete")
        decoded_tensor = self.decode(encoded_tensor)
        print("Decoding complete")
        return decoded_tensor

sf8 = Superfloat(8)  # Create an SF8 superfloat type

class QuantizedLlamaModel(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, sf8_quantizer: Superfloat):
        super(QuantizedLlamaModel, self).__init__()
        self.base_model = base_model
        self.sf8_quantizer = sf8_quantizer
        self.apply_gradient_hooks()

    def apply_gradient_hooks(self):
        # Register a hook to quantize gradients after backward pass
        for param in self.base_model.parameters():
            param.register_hook(lambda grad: self.sf8_quantizer.tensor_quantize(grad))

    def forward(self, x):
        # Quantize activations and parameters during forward pass
        x = self.sf8_quantizer.tensor_quantize(x)
        for layer in self.base_model.children():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data = self.sf8_quantizer.tensor_quantize(layer.weight.data)
            x = self.sf8_quantizer.tensor_quantize(layer(x))
        return x

from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
model = LlamaForCausalLM.from_pretrained(model_name, cache_dir='./', token='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')
model = model.to(torch.bfloat16).to(device)

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, cache_dir='./', token='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')

tokenizer.pad_token = tokenizer.eos_token

def quantize_model(model, sf_type):
    for name, param in model.named_parameters():
        print(name, len(param))
        quantized_param = sf_type.tensor_quantize(param)
        param.data = quantized_param.data
    return model

quantized = quantize_model(model, sf8)
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
        torch.save(quantized.state_dict(), f"sf8_pile_epoch{epoch+1}")
    print(f"Epoch {epoch + 1} completed with average loss: {epoch_loss:.4f}")