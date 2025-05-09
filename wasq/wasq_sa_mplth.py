import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import copy
import numpy as np
import math
from collections import defaultdict

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

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantizes a tensor to Superfloat format with gradient preservation"""
        with torch.no_grad():
            scale = self.max_val / tensor.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            quantized = torch.clamp(tensor * scale, -self.max_val, self.max_val)
            return quantized / scale

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

class SA_MPLTH_Trainer:
    def __init__(self, model, sf_quantizer, tokenizer, config):
        self.device = device
        self.sf_quantizer = sf_quantizer
        self.model = model.to(device=self.device, dtype=sf_quantizer.float_type)
        self.tokenizer = tokenizer
        self.config = config
        
        # SA-MPLTH specific parameters
        self.initial_temp = config.get('initial_temp', 1.0)
        self.final_temp = config.get('final_temp', 0.01)
        self.temp_decay = config.get('temp_decay', 0.95)
        self.current_temp = self.initial_temp
        
        self.original_model_state = copy.deepcopy(self.model.state_dict())
        self.winning_tickets = defaultdict(dict)
        self.subnetwork_scores = {}
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-5),
            eps=config.get('optimizer_eps', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.get('mixed_precision', True))

    def prepare_dataset(self, max_length=512):
        """Prepares dataset with proper tokenization"""
        dataset = Dataset.from_parquet('train.parquet')
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
        
        return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    def create_dataloader(self, dataset, batch_size=4):
        """Creates a DataLoader with proper batching"""
        def collate_fn(batch):
            return {
                'input_ids': torch.stack([torch.tensor(x['input_ids']) for x in batch]),
                'attention_mask': torch.stack([torch.tensor(x['attention_mask']) for x in batch])
            }
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def anneal_temperature(self, epoch, total_epochs):
        """Exponential temperature decay for simulated annealing"""
        self.current_temp = max(
            self.final_temp,
            self.initial_temp * (self.temp_decay ** epoch)
        )
        return self.current_temp

    def generate_subnetwork_masks(self, model, num_subnetworks=3):
        """Generates multiple complementary subnetworks using magnitude pruning"""
        masks = {}
        weights = {n: p.data.abs() for n, p in model.named_parameters() if p.dim() > 1}
        
        for i in range(num_subnetworks):
            masks[i] = {}
            for name, weight in weights.items():
                # Different pruning thresholds for each subnetwork
                threshold = torch.quantile(
                    weight.flatten(),
                    q=self.config['pruning_rates'][i % len(self.config['pruning_rates'])]
                )
                masks[i][name] = (weight > threshold).float()
        
        return masks

    def apply_quantization_noise(self, model, masks, subnetwork_idx):
        """Applies simulated annealing noise to weights"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in masks[subnetwork_idx]:
                    noise = torch.randn_like(param) * self.current_temp
                    param.data = param.data * masks[subnetwork_idx][name] + noise * (1 - masks[subnetwork_idx][name])
                    param.data = self.sf_quantizer.quantize(param.data)

    def evaluate_subnetwork(self, model, dataloader, subnetwork_idx):
        """Evaluates subnetwork performance on validation set"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        # Apply the subnetwork mask before evaluation
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.winning_tickets[subnetwork_idx]:
                    param.data.copy_(self.winning_tickets[subnetwork_idx][name])
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input_ids'].to(self.device)
                masks = batch['attention_mask'].to(self.device)
                
                outputs = model(input_ids=inputs, attention_mask=masks)
                logits = outputs.logits
                
                # Shift for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs[..., 1:].contiguous()
                shift_mask = masks[..., 1:].contiguous()
                
                loss = self.loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                total_loss += loss.item() * shift_mask.sum().item()
                total_tokens += shift_mask.sum().item()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        print(f"Subnetwork {subnetwork_idx} validation perplexity: {perplexity:.2f}")
        return perplexity

    def train_subnetwork(self, model, dataloader, val_dataloader, masks, subnetwork_idx, num_epochs=3):
        """Trains a single subnetwork with quantization-aware training"""
        best_perplexity = float('inf')
        
        for epoch in range(num_epochs):
            self.anneal_temperature(epoch, num_epochs)
            model.train()
            epoch_loss = 0
            
            progress_bar = tqdm(dataloader, desc=f"Subnet {subnetwork_idx} Epoch {epoch+1}")
            
            for batch in progress_bar:
                self.optimizer.zero_grad()
                
                inputs = batch['input_ids'].to(self.device)
                masks = batch['attention_mask'].to(self.device)
                
                with torch.cuda.amp.autocast(device_type=device, enabled=self.config.get('mixed_precision', True)):
                    outputs = model(input_ids=inputs, attention_mask=masks)
                    logits = outputs.logits
                    
                    # Shift for next-token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs[..., 1:].contiguous()
                    shift_mask = masks[..., 1:].contiguous()
                    
                    loss = self.loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                
                self.scaler.scale(loss).backward()
                
                # Apply gradient masking for the subnetwork
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in masks[subnetwork_idx]:
                            param.grad *= masks[subnetwork_idx][name]
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "temp": f"{self.current_temp:.4f}"})
            
            # Evaluate on validation set after each epoch
            current_perplexity = self.evaluate_subnetwork(model, val_dataloader, subnetwork_idx)
            if current_perplexity < best_perplexity:
                best_perplexity = current_perplexity
                torch.save(model.state_dict(), f"best_subnet_{subnetwork_idx}.pth")
        
        return best_perplexity

    def merge_subnetworks(self, model, subnetworks):
        """Merges multiple subnetworks using learned weights"""
        merged_state = copy.deepcopy(self.original_model_state)
        
        for name in merged_state:
            if any(name in sub for sub in subnetworks.values()):
                weights = [sub[name] for sub in subnetworks.values() if name in sub]
                # Weighted average based on subnetwork scores (better subnetworks get more weight)
                weights = torch.stack(weights)
                scores = torch.tensor([1/self.subnetwork_scores[i] for i in range(len(weights))])
                merged_state[name] = (weights * scores.view(-1, 1, 1)).sum(dim=0) / scores.sum()
        
        model.load_state_dict(merged_state)
        return model

    def train(self):
        """Main training loop for SA-MPLTH"""
        # Prepare data
        tokenized_dataset = self.prepare_dataset()
        train_dataloader = self.create_dataloader(tokenized_dataset)
        val_dataloader = self.create_dataloader(tokenized_dataset, batch_size=8)  # Smaller batch for validation
        
        # Generate initial subnetworks
        subnetworks = self.generate_subnetwork_masks(self.model)
        
        # Phase 1: Train subnetworks independently
        for subnetwork_idx in subnetworks:
            print(f"\nTraining Subnetwork {subnetwork_idx}")
            
            # Create a fresh copy of the model
            submodel = copy.deepcopy(self.model)
            submodel.load_state_dict(self.original_model_state)
            
            # Apply quantization and noise
            self.apply_quantization_noise(submodel, subnetworks, subnetwork_idx)
            
            # Train the subnetwork with validation
            perplexity = self.train_subnetwork(
                submodel,
                train_dataloader,
                val_dataloader,  # Now passing validation dataloader
                subnetworks,
                subnetwork_idx,
                num_epochs=self.config.get('subnetwork_epochs', 3)
            )
            
            # Store results
            self.winning_tickets[subnetwork_idx] = submodel.state_dict()
            self.subnetwork_scores[subnetwork_idx] = perplexity
            print(f"Subnetwork {subnetwork_idx} achieved perplexity: {perplexity:.2f}")
        
        # Phase 2: Merge and fine-tune
        print("\nMerging subnetworks...")
        merged_model = self.merge_subnetworks(self.model, self.winning_tickets)
        
        # Final fine-tuning of merged model with validation
        print("Final fine-tuning...")
        best_perplexity = self.train_subnetwork(
            merged_model,
            train_dataloader,
            val_dataloader,  # Using validation for final model
            {},  # No masking for final training
            "final",
            num_epochs=self.config.get('final_epochs', 5)
        )
        
        print(f"\nTraining complete! Final validation perplexity: {best_perplexity:.2f}")
        return merged_model

def main():
    # Initialize model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Superfloat configuration
    sf_quantizer = Superfloat(bits=8)  # Using 8-bit superfloat
    
    # Training configuration
    config = {
        "pruning_rates": [0.7, 0.8, 0.9],  # Different sparsity levels
        "initial_temp": 1.0,
        "final_temp": 0.01,
        "temp_decay": 0.95,
        "learning_rate": 1e-5,
        "optimizer_eps": 1e-4,
        "weight_decay": 0.01,
        "mixed_precision": True,
        "subnetwork_epochs": 3,
        "final_epochs": 5,
        "batch_size": 4,
        "max_length": 512
    }
    
    # Initialize and run trainer
    trainer = SA_MPLTH_Trainer(model, sf_quantizer, tokenizer, config)
    final_model = trainer.train()
    
    # Save final model
    torch.save(final_model.state_dict(), "sa_mplth_final_model.pth")

if __name__ == "__main__":
    main()