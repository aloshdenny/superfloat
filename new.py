import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
import gc
import numpy as np
from sklearn.cluster import KMeans

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

# This class tracks activations and finds clusters of parameters with high correlations
class ActivationTracker:
    def __init__(self, model, sf_quantizer, device):
        self.model = model
        self.sf_quantizer = sf_quantizer
        self.device = device
        self.activations = {}
        self.parameter_names = []
        self.apply_hooks()

    def apply_hooks(self):
        """Register hooks to track activations."""
        def hook_fn(module, input, output):
            # Record activations (we can store them per layer)
            self.activations[module] = output.detach().cpu().numpy()
            self.parameter_names.append(module.__class__.__name__)

        # Register hooks on each layer
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
                layer.register_forward_hook(hook_fn)

    def compute_activation_correlation(self):
        """Compute correlation of activations across different layers/neurons."""
        activations_list = []
        for layer_name, activation in self.activations.items():
            activations_list.append(activation.flatten())  # Flatten to 1D to compute correlations

        activations_matrix = np.stack(activations_list, axis=0)
        correlation_matrix = np.corrcoef(activations_matrix)
        return correlation_matrix

    def cluster_activations(self, correlation_matrix, num_clusters=5):
        """Use KMeans clustering to group parameters with high correlation in their activations."""
        # Apply KMeans clustering to the correlation matrix
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(correlation_matrix)
        return kmeans.labels_

    def target_and_retrain(self, labels):
        """Target the critical parameters from the clustered activations and retrain."""
        critical_params = []
        for layer_name, label in zip(self.parameter_names, labels):
            if label == 0:  # For instance, consider cluster 0 as critical
                critical_params.append(layer_name)
        return critical_params

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

# Initialize quantization
sf = Superfloat(11)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
model.to(sf.float_type).to(device)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", cache_dir='./', token='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')
tokenizer.pad_token = tokenizer.eos_token

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

# Initialize tracker for activation and clustering
tracker = ActivationTracker(model, sf, device)

# Training loop with tracking and retraining clusters
def training_loop(model, tracker, num_epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, eps=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            target = input_ids[:, 1:].contiguous()
            logits = logits[:, :-1].contiguous()

            # Calculate loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))

            # Backward pass
            loss.backward()

            epoch_loss += loss.item()

            # Every few steps, compute activations and update
            if (step + 1) % 100 == 0:
                correlation_matrix = tracker.compute_activation_correlation()
                labels = tracker.cluster_activations(correlation_matrix)
                critical_params = tracker.target_and_retrain(labels)

                # Retrain critical parameters only
                for name, param in model.named_parameters():
                    if name in critical_params:
                        optimizer.step()

                optimizer.zero_grad()

        print(f"Epoch {epoch + 1} completed with loss: {epoch_loss / len(train_dataloader)}")

# Start the training loop
training_loop(model, tracker)