import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import gc
import os
import numpy as np
import copy

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
        mantissa = (torch.abs(clipped_value) * (2**self.mantissa_bits - 1) / self.max_val).floor().to(torch.int32)
        sign = (clipped_value < 0).to(torch.int32)
        return (mantissa | (sign << self.mantissa_bits)).to(torch.int32)

    def decode(self, encoded_value: torch.Tensor) -> torch.Tensor:
        """Decodes a tensor of encoded superfloat values to regular floats."""
        mantissa = encoded_value & ((1 << self.mantissa_bits) - 1)
        sign = (encoded_value >> self.mantissa_bits) & 1
        decoded_value = (mantissa.to(self.float_type) / (2**self.mantissa_bits - 1)) * self.max_val
        return decoded_value * (2 * sign - 1)

    def tensor_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantizes a tensor to the superfloat format and decodes back."""
        encoded_tensor = self.encode(tensor)
        decoded_tensor = self.decode(encoded_tensor)
        return decoded_tensor

class LotteryTicketTrainer:
    def __init__(self, model, sf_quantizer, tokenizer, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sf_quantizer = sf_quantizer
        self.model = model.to(device=self.device, dtype=sf_quantizer.float_type)
        self.tokenizer = tokenizer
        self.config = config
        self.original_model_state = copy.deepcopy(self.model.state_dict())
        self.winning_tickets = {}
        self.pruning_rate = config.get('pruning_rate', 0.2)
        self.pruning_iterations = config.get('pruning_iterations', 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 1e-5), eps=config.get('optimizer_eps', 1e-4))
        self.loss_fn = nn.CrossEntropyLoss()

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
                mask = pruning_masks[name]
                original_init = self.original_model_state[name]
                param.data = original_init * mask

    def activation_magnitude_analysis(self):
        # Placeholder: Implement activation magnitude analysis after each forward pass
        layer_activation_changes = {}
        for name, param in self.model.named_parameters():
            # Analyze activation differences between the original and quantized models (per layer)
            pass  # Here you would track activations before and after quantization
        return layer_activation_changes

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
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    target = input_ids[:, 1:].contiguous()
                    logits = logits[:, :-1].contiguous()
                    
                    loss = self.loss_fn(
                        logits.view(-1, logits.size(-1)), 
                        target.view(-1)
                    ) / accumulation_steps
                    
                    loss.backward()
                    epoch_loss += loss.item() * accumulation_steps
                    
                    if (step + 1) % accumulation_steps == 0:
                        for param in self.model.parameters():
                            param.data = torch.clamp(
                                param.data, 
                                min=-self.sf_quantizer.max_val, 
                                max=self.sf_quantizer.max_val
                            )
                        
                        torch.nn.utils.clip_grad_value_(
                            self.model.parameters(), 
                            clip_value=self.sf_quantizer.max_val
                        )
                        
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
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

def main():
    # Load the pre-trained model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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

if __name__ == "__main__":
    main()
