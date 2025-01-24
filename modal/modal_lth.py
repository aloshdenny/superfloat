import modal

# Create a Modal image with the required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "huggingface_hub",
        "requests",
    )
    .apt_install("gcc", "python3-dev")  # Add necessary system libraries if needed
)

app = modal.App("superfloat-LTH")

@app.function(gpu="H100", image=image, timeout=86400)
def train_and_upload():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader
    from datasets import Dataset
    from tqdm import tqdm
    import os
    import numpy as np
    import copy
    import requests

    # URL of the dataset
    url = "https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated/resolve/main/data/train-00000-of-01650-f70471ee3deb09c0.parquet"

    # Local file paths
    downloaded_file = "train-00000-of-01650-f70471ee3deb09c0.parquet"
    final_file_name = "train.parquet"

    # Download the dataset
    if not os.path.exists(final_file_name):  # Check if the file is already downloaded
        print(f"Downloading dataset from {url}...")
        response = requests.get(url, stream=True)
        with open(downloaded_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to {downloaded_file}.")

        # Rename the file
        os.rename(downloaded_file, final_file_name)
        print(f"Renamed to {final_file_name}.")
    else:
        print(f"{final_file_name} already exists. Skipping download.")

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

        def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
            """Quantizes a tensor to Superfloat format."""
            # Per-channel scaling for dynamic range
            scale = self.max_val / tensor.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            quantized = torch.clamp(tensor * scale, -self.max_val, self.max_val).round()
            return quantized / scale  # Dequantize for inference

    class LotteryTicketTrainer:
        def __init__(self, model, sf_quantizer, tokenizer, config):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.sf_quantizer = sf_quantizer
            self.model = model.to(device=self.device)
            self.tokenizer = tokenizer
            self.config = config
            self.original_model_state = copy.deepcopy(self.model.state_dict())
            self.winning_tickets = {}
            self.pruning_rate = config.get('pruning_rate', 0.2)
            self.pruning_iterations = config.get('pruning_iterations', 3)
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 1e-5), eps=config.get('optimizer_eps', 1e-4))
            self.loss_fn = nn.CrossEntropyLoss()
            self.scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

        def prepare_dataset(self, max_length=2):
            dataset = Dataset.from_parquet('train.parquet')
            
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
            return tokenized_dataset

        def create_dataloader(self, dataset, batch_size=4):
            def collate_fn(batch):
                input_ids = torch.stack([torch.tensor(example['input_ids']) for example in batch])
                attention_mask = torch.stack([torch.tensor(example['attention_mask']) for example in batch])
                return {'input_ids': input_ids, 'attention_mask': attention_mask}
            
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        def magnitude_based_pruning(self):
            pruning_masks = {}
            
            for name, param in self.model.named_parameters():
                if len(param.shape) > 1:  # Only prune weight matrices
                    weight_abs = torch.abs(param.data)
                    flat_weights = weight_abs.view(-1)
                    k = int(flat_weights.numel() * self.pruning_rate)
                    threshold = torch.topk(flat_weights, k, largest=False).values.max()
                    mask = (weight_abs > threshold).float()
                    pruning_masks[name] = mask
                    param.data *= mask
            
            return pruning_masks

        def reset_to_winning_ticket(self, pruning_masks):
            for name, param in self.model.named_parameters():
                if name in pruning_masks:
                    # Reset to original initialization, then apply mask
                    param.data.copy_(self.original_model_state[name])
                    param.data *= pruning_masks[name]

        def activation_magnitude_analysis(self):
            # Compare activations between original and quantized models
            with torch.no_grad():
                original_activations = self.get_activations(self.original_model_state)
                quantized_activations = self.get_activations(self.model.state_dict())
                return self.compute_layerwise_differences(original_activations, quantized_activations)

        def get_activations(self, model_state):
            # Placeholder: Implement forward pass to collect activations
            activations = {}
            for name, param in model_state.items():
                if len(param.shape) > 1:
                    activations[name] = torch.mean(torch.abs(param)).item()
            return activations

        def compute_layerwise_differences(self, original_activations, quantized_activations):
            differences = {}
            for name in original_activations:
                differences[name] = abs(original_activations[name] - quantized_activations[name])
            return differences

        def fine_tune_based_on_activations(self, layer_activation_changes):
            # Fine-tune layers with significant activation change
            for layer, change in layer_activation_changes.items():
                if change > self.config.get('activation_threshold', 0.1):
                    # Fine-tune or adjust this layer specifically
                    pass  # Fine-tune layer weights based on magnitude analysis

        def train(self):
            tokenized_dataset = self.prepare_dataset()
            dataloader = self.create_dataloader(tokenized_dataset)
            
            num_epochs = self.config.get('num_epochs', 3)
            accumulation_steps = self.config.get('accumulation_steps', 32)
            best_loss = float('inf')
            
            for iteration in range(self.pruning_iterations):
                print(f"\nPruning Iteration {iteration + 1}/{self.pruning_iterations}")
                
                for epoch in range(num_epochs):
                    self.model.train()
                    epoch_loss = 0.0
                    
                    epoch_iterator = tqdm(
                        enumerate(dataloader),
                        total=len(dataloader),
                        desc=f"Iteration {iteration + 1}, Epoch {epoch + 1}"
                    )
                    
                    for step, batch in epoch_iterator:
                        input_ids = batch['input_ids'].to(device=self.device, dtype=torch.long)
                        attention_mask = batch['attention_mask'].to(device=self.device, dtype=torch.long)
                        
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                            logits = outputs.logits
                            
                            target = input_ids[:, 1:].contiguous()
                            logits = logits[:, :-1].contiguous()
                            
                            loss = self.loss_fn(
                                logits.view(-1, logits.size(-1)), 
                                target.view(-1)
                            ) / accumulation_steps
                        
                        self.scaler.scale(loss).backward()
                        epoch_loss += loss.item() * accumulation_steps
                        
                        if (step + 1) % accumulation_steps == 0:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                            
                            # Clamp parameters within Superfloat range
                            for param in self.model.parameters():
                                param.data = torch.clamp(
                                    param.data, 
                                    min=-self.sf_quantizer.max_val, 
                                    max=self.sf_quantizer.max_val
                                )
                            
                            epoch_iterator.set_postfix({"Loss": f"{loss.item() * accumulation_steps:.4f}"})
                    
                    epoch_loss /= len(dataloader)
                    print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")
                    
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        torch.save(
                            self.model.state_dict(), 
                            f"sf{self.sf_quantizer.bits}_iteration{iteration+1}_epoch{epoch+1}_best.pth"
                        )
                
                pruning_masks = self.magnitude_based_pruning()
                self.reset_to_winning_ticket(pruning_masks)
                
                # After pruning, perform activation analysis and fine-tuning
                layer_activation_changes = self.activation_magnitude_analysis()
                self.fine_tune_based_on_activations(layer_activation_changes)
                
                torch.save(self.model.state_dict(), f"sf{self.sf_quantizer.bits}_winning_ticket_iteration{iteration+1}.pth")

    # Load the pre-trained model and tokenizer
    model_name = "Qwen/Qwen2-0.5B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Initialize the Superfloat quantizer
    sf_quantizer = Superfloat(bits=11)  # You can experiment with different bit-widths
    
    # Configuration settings
    config = {
        "pruning_rate": 0.2,
        "pruning_iterations": 3,
        "learning_rate": 1e-5,
        "optimizer_eps": 1e-4,
        "num_epochs": 3,
        "accumulation_steps": 32,
        "activation_threshold": 0.1  # You can adjust this threshold
    }
    
    # Instantiate the trainer
    trainer = LotteryTicketTrainer(model, sf_quantizer, tokenizer, config)
    
    # Train the model with LTH + Superfloat quantization
    trainer.train()

@app.local_entrypoint()
def main():
    train_and_upload.remote()