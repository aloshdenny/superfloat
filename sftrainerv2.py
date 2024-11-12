import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
import gc

class Superfloat:
    def __init__(self, bits: int):
        assert 4 <= bits <= 16, "Superfloat bitwidth must be between 4 and 16."
        self.bits = bits
        self.mantissa_bits = bits - 1
        self.max_val = 1 - 2**-self.mantissa_bits  # Precompute max representable value

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        """Encodes a tensor of values into the superfloat format with optimized operations."""
        clipped_value = torch.clamp(value, min=-self.max_val, max=self.max_val)
        out_of_range = (value.abs() > self.max_val)

        # Calculate mantissa representation element-wise
        mantissa = (torch.abs(clipped_value) * (2**self.mantissa_bits - 1) / self.max_val).floor().to(torch.int32)

        # Create the superfloat representation (1 bit for sign and mantissa bits)
        sign = (clipped_value < 0).to(torch.int32)
        return (mantissa | (sign << self.mantissa_bits)).to(torch.int32), out_of_range

    def decode(self, encoded_value: torch.Tensor) -> torch.Tensor:
        """Decodes a tensor of encoded superfloat values to regular floats."""
        mantissa = encoded_value & ((1 << self.mantissa_bits) - 1)
        sign = (encoded_value >> self.mantissa_bits) & 1
        decoded_value = (mantissa.to(torch.bfloat16) / (2**self.mantissa_bits - 1)) * self.max_val
        return decoded_value * (2 * sign - 1)

    def tensor_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantizes a tensor to the superfloat format, preserving the tensor's shape."""
        encoded_tensor, out_of_range = self.encode(tensor)
        decoded_tensor = self.decode(encoded_tensor)
        return decoded_tensor, out_of_range

sf8 = Superfloat(8)

class QuantizedLlamaModel(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, sf8_quantizer: Superfloat):
        super(QuantizedLlamaModel, self).__init__()
        self.base_model = base_model
        self.sf8_quantizer = sf8_quantizer
        self.apply_gradient_hooks()

    def apply_gradient_hooks(self):
        for param in self.base_model.parameters():
            def hook(grad, param=param):
                _, out_of_range = self.sf8_quantizer.tensor_quantize(param)
                grad = grad * out_of_range.to(grad.dtype)  # Mask to allow gradients only on out-of-range params
                return grad
            param.register_hook(hook)

    def forward(self, x):
        x, _ = self.sf8_quantizer.tensor_quantize(x)
        for layer in self.base_model.children():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data, _ = self.sf8_quantizer.tensor_quantize(layer.weight.data)
            x = layer(x)
            x, _ = self.sf8_quantizer.tensor_quantize(x)
        return x

# Initialize model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "meta-llama/Llama-3.2-1B"
model = LlamaForCausalLM.from_pretrained(model_name, cache_dir='./', token='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')
model = model.to(torch.bfloat16).to(device)

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, cache_dir='./', token='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')
tokenizer.pad_token = tokenizer.eos_token

# Quantize Model Weights Selectively
def quantize_model(model, sf_type):
    for name, param in model.named_parameters():
        quantized_param, _ = sf_type.tensor_quantize(param)
        param.data = quantized_param.data
    return model

quantized = quantize_model(model, sf8)
del model
torch.cuda.empty_cache()
gc.collect()

# Prepare Dataset
def prepare_dataset(tokenizer, max_length=512):
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
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
    if epoch_loss < best_loss:
        torch.save(quantized.state_dict(), f"sf8_pile_epoch{epoch+1}")
    print(f"Epoch {epoch + 1} completed with average loss: {epoch_loss:.4f}")