import modal

# Create a Modal image with the required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers",
        "torch",
        "datasets",
        "tqdm",
        "huggingface_hub",
        "tokenizers",
        "tensorboard",

    )
    .apt_install("gcc", "python3-dev")  # Add necessary system libraries if needed
)

app = modal.App("superfloat-train-gpt")

@app.function(gpu="H100", image=image, timeout=86400)
def train_and_upload():

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.nn import TransformerEncoder
    import math
    from tqdm import tqdm
    import torch.nn.functional as F
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from torch.utils.tensorboard import SummaryWriter
    import os

    class SuperFloatTensor:
        def __init__(self, data):
            # Use in-place operations to avoid temporary copies
            self.data = data.clamp_(-0.9999999, 0.9999999)  # _ suffix for in-place
            
        @staticmethod
        def _clamp(x):
            return x.clamp_(-0.9999999, 0.9999999)
            
        def __matmul__(self, other):
            # Use torch's built-in matmul with fused operations
            return SuperFloatTensor(torch.matmul(self.data, other.data).clamp_(-0.9999999, 0.9999999))
        
    class ConstrainedGELU(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x):
            # Original GELU approximation
            gelu = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
            # Constrain output to (-1, 1)
            return torch.tanh(gelu) * 0.99

    class ConstrainedLinear(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            # Initialize weights in (-1, 1)
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.bias = nn.Parameter(torch.empty(out_features))
            self.reset_parameters()
            
        def reset_parameters(self):
            # Xavier initialization scaled to (-1, 1)
            nn.init.xavier_uniform_(self.weight, gain=0.99)
            nn.init.uniform_(self.bias, -0.99, 0.99)
            
        def forward(self, input):
            # Input should already be in (-1, 1)
            output = torch.nn.functional.linear(input, self.weight, self.bias)
            return torch.tanh(output) * 0.99  # Keep in (-1, 1)
        
    class ConstrainedAttention(nn.Module):
        def __init__(self, embed_dim, num_heads, batch_first=True, norm_first=True):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.batch_first = batch_first
            self.norm_first = norm_first
            
            assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
            
            # Combined QKV projection
            self.qkv = ConstrainedLinear(embed_dim, embed_dim * 3)
            self.proj = ConstrainedLinear(embed_dim, embed_dim)
            
        def forward(self, x, attn_mask=None, key_padding_mask=None, is_causal=False):
            B, N, C = x.shape
            
            # Project QKV
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
            
            # Scaled dot-product attention
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            
            # Apply causal mask if needed
            if is_causal:
                causal_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=x.device), diagonal=1)
                attn = attn.masked_fill(causal_mask, float('-inf'))
            
            # Apply attention mask
            if attn_mask is not None:
                attn = attn + attn_mask
                
            # Apply key padding mask
            if key_padding_mask is not None:
                attn = attn.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf')
                )
            
            # Constrain and softmax
            attn = torch.tanh(attn) * 0.99
            attn = torch.softmax(attn, dim=-1)
            
            # Apply to values
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            
            # Final projection with constraints
            x = self.proj(x)
            return torch.tanh(x) * 0.99
        
    class ConstrainedTransformerEncoderLayer(nn.Module):
        def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                    activation="gelu", layer_norm_eps=1e-5, batch_first=True, 
                    norm_first=True):
            super().__init__()
            self.d_model = d_model
            self.norm_first = norm_first
            
            # Self attention with constraints
            self.self_attn = ConstrainedAttention(d_model, nhead, batch_first)
            
            # Feedforward layers
            self.linear1 = ConstrainedLinear(d_model, dim_feedforward)
            self.linear2 = ConstrainedLinear(dim_feedforward, d_model)
            
            # Normalization layers
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            
            # Dropout layers
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
            
            # Activation
            self.activation = ConstrainedGELU() if activation == "gelu" else nn.Tanh()

        def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
            x = src
            if self.norm_first:
                # Pre-LN architecture
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
                x = x + self._ff_block(self.norm2(x))
            else:
                # Post-LN architecture
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal))
                x = self.norm2(x + self._ff_block(x))
            return x

        def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
            x = self.self_attn(x, attn_mask=attn_mask, 
                            key_padding_mask=key_padding_mask,
                            is_causal=is_causal)
            return self.dropout1(x)

        def _ff_block(self, x):
            x = self.linear2(self.dropout2(self.activation(self.linear1(self.dropout3(x)))))
            return x
        
    class ConstrainedAdam(optim.Adam):
        def step(self, closure=None):
            loss = super().step(closure)
            
            # Clamp all parameters after update
            for group in self.param_groups:
                for p in group['params']:
                    p.data = torch.clamp(p.data, -0.99, 0.99)
            return loss

    def constrained_loss_fn(logits, targets):
        # Example for cross-entropy - need to constrain logits
        logits = torch.tanh(logits) * 0.99
        return F.cross_entropy(logits, targets)

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=512, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
            
        def forward(self, x):
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(torch.tanh(x) * 0.99)
        
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, seq_length, max_length=None):
            """
            Args:
                encodings: List of tokenized text
                seq_length: Length of each sequence sample
                max_length: Maximum length to truncate/pad to (defaults to seq_length)
            """
            self.encodings = encodings
            self.seq_length = seq_length
            self.max_length = max_length if max_length is not None else seq_length
            
        def __len__(self):
            return len(self.encodings) - self.seq_length
            
        def __getitem__(self, idx):
            # Get sequence chunk of length seq_length + 1 (for target)
            chunk = self.encodings[idx:idx + self.seq_length + 1]
            
            # Pad or truncate if necessary
            if len(chunk) < self.seq_length + 1:
                chunk = chunk + [0] * (self.seq_length + 1 - len(chunk))  # 0 is padding token
            elif len(chunk) > self.seq_length + 1:
                chunk = chunk[:self.seq_length + 1]
                
            return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)
        
    def prepare_datasets(config):
        # Load WikiText-2 dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        
        # Train tokenizer on the training data
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        
        def batch_iterator():
            batch_size = 1000
            for i in range(0, len(dataset['train']['text']), batch_size):
                yield dataset['train']['text'][i:i + batch_size]
        
        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        
        # Tokenize all datasets
        def tokenize(text):
            return tokenizer.encode(text).ids
        
        train_texts = [tokenize(text) for text in dataset['train']['text'] if text.strip()]
        val_texts = [tokenize(text) for text in dataset['validation']['text'] if text.strip()]
        test_texts = [tokenize(text) for text in dataset['test']['text'] if text.strip()]
        
        # Flatten and create datasets
        train_encodings = [tok for text in train_texts for tok in text]
        val_encodings = [tok for text in val_texts for tok in text]
        test_encodings = [tok for text in test_texts for tok in text]
        
        train_dataset = TextDataset(train_encodings, config['seq_length'])
        val_dataset = TextDataset(val_encodings, config['seq_length'])
        test_dataset = TextDataset(test_encodings, config['seq_length'])
        
        return train_dataset, val_dataset, test_dataset, tokenizer.get_vocab_size()

    def calculate_perplexity(loss):
        return math.exp(loss)

    class SuperFloatConstrainedTransformer(nn.Module):
        def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, 
                    dim_feedforward=2048, dropout=0.1, max_seq_len=512,
                    activation="gelu", layer_norm_eps=1e-5):
            super().__init__()
            self.d_model = d_model
            self.num_layers = num_layers
            self.embed_scale = 0.99 * math.sqrt(d_model)
            
            # Embedding and positional encoding
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
            
            # Create encoder layers
            self.layers = nn.ModuleList([
                ConstrainedTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                    batch_first=True,
                    norm_first=True
                )
                for _ in range(num_layers)
            ])
            
            # Final layer norm and output projection
            self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.fc_out = ConstrainedLinear(d_model, vocab_size)
            
            self._init_weights()
            
        def _init_weights(self):
            """Initialize weights like modern LLMs but constrained to (-1, 1)"""
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
            self.embedding.weight.data = torch.clamp(self.embedding.weight.data, -0.99, 0.99)
            
            for name, p in self.named_parameters():
                if 'weight' in name and p.dim() > 1:
                    if 'fc_out' in name:
                        nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.num_layers))
                    else:
                        nn.init.xavier_uniform_(p, gain=0.99/math.sqrt(2))
                    p.data = torch.clamp(p.data, -0.99, 0.99)
                elif 'bias' in name:
                    nn.init.constant_(p, 0.0)
                    
        def forward(self, src, src_mask=None, src_key_padding_mask=None):
            # Input embeddings with constraints
            src = self.embedding(src) * self.embed_scale
            src = torch.tanh(src) * 0.99
            
            # Positional encoding
            src = self.pos_encoder(src)
            
            # Transformer layers
            for layer in self.layers:
                src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            
            # Final normalization and projection
            output = self.norm(src)
            output = self.fc_out(output)
            return torch.tanh(output) * 0.99
        
    def train_model(config):
        # Setup device and logging
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        writer = SummaryWriter(log_dir=config['log_dir'])
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset, vocab_size = prepare_datasets(config)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
        
        # Initialize model
        model = SuperFloatConstrainedTransformer(
            vocab_size=vocab_size,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            max_seq_len=config['seq_length'],
            activation=config['activation']
        ).to(device)
        
        # Optimizer and scheduler
        optimizer = ConstrainedAdam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['lr'],
            steps_per_epoch=len(train_loader),
            epochs=config['epochs']
        )
        
        # criterion = nn.CrossEntropyLoss(ignore_index=0)
        criterion = constrained_loss_fn
        
        # Training loop
        best_val_loss = float('inf')
        global_step = 0
        
        for epoch in range(config['epochs']):
            model.train()
            train_loss = 0
            train_perplexity = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch_idx, (src, tgt) in enumerate(progress_bar):
                src, tgt = src.to(device), tgt.to(device)
                
                optimizer.zero_grad()
                output = model(src)
                
                loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
                scheduler.step()
                
                # Calculate perplexity
                perplexity = calculate_perplexity(loss.item())
                
                # Update metrics
                train_loss += loss.item()
                train_perplexity += perplexity
                
                # Log step metrics
                writer.add_scalar('train/loss_step', loss.item(), global_step)
                writer.add_scalar('train/perplexity_step', perplexity, global_step)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'ppl': perplexity,
                    'lr': scheduler.get_last_lr()[0]
                })
                
                global_step += 1
                
                # Validation at regular intervals
                if batch_idx % config['eval_steps'] == 0:
                    val_loss, val_perplexity = evaluate(model, val_loader, criterion, device, vocab_size)
                    writer.add_scalar('val/loss', val_loss, global_step)
                    writer.add_scalar('val/perplexity', val_perplexity, global_step)
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), f"{config['log_dir']}/best_model.pth")
                    
                    model.train()
            
            # Epoch metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_train_perplexity = train_perplexity / len(train_loader)
            
            # Full validation at end of epoch
            val_loss, val_perplexity = evaluate(model, val_loader, criterion, device, vocab_size)
            
            print(f"\nEpoch {epoch+1}:")
            print(f"Train Loss: {avg_train_loss:.4f} | Train PPL: {avg_train_perplexity:.2f}")
            print(f"Val Loss: {val_loss:.4f} | Val PPL: {val_perplexity:.2f}")
            
            writer.add_scalar('train/loss_epoch', avg_train_loss, epoch)
            writer.add_scalar('train/perplexity_epoch', avg_train_perplexity, epoch)
            writer.add_scalar('val/loss_epoch', val_loss, epoch)
            writer.add_scalar('val/perplexity_epoch', val_perplexity, epoch)
        
        # Final test evaluation
        test_loss, test_perplexity = evaluate(model, test_loader, criterion, device, vocab_size)
        print(f"\nFinal Test Results:")
        print(f"Test Loss: {test_loss:.4f} | Test PPL: {test_perplexity:.2f}")
        writer.add_scalar('test/loss', test_loss, global_step)
        writer.add_scalar('test/perplexity', test_perplexity, global_step)
        
        writer.close()

        # Save the final model
        model_path = f"{config['log_dir']}/final_model"
        os.makedirs(model_path, exist_ok=True)
        torch.save(model.state_dict(), f"{model_path}/pytorch_model.bin")
        
        # Save config
        with open(f"{model_path}/config.json", "w") as f:
            import json
            json.dump(config, f)
        
        # Upload to Hugging Face Hub using CLI
        os.system(
            f"huggingface-cli upload aoxo/superfloat-gpt {model_path}/* --token='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'"
        )
        return model

    def evaluate(model, data_loader, criterion, device, vocab_size):
        model.eval()
        total_loss = 0
        total_perplexity = 0
        
        with torch.no_grad():
            for src, tgt in data_loader:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src)
                loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
                total_loss += loss.item()
                total_perplexity += calculate_perplexity(loss.item())
        
        avg_loss = total_loss / len(data_loader)
        avg_perplexity = total_perplexity / len(data_loader)
        return avg_loss, avg_perplexity

    # Configuration
    config = {
        'd_model': 512,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'seq_length': 2048,
        'activation': 'gelu',
        'batch_size': 32,
        'lr': 5e-4,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'epochs': 10,
        'eval_steps': 200,
        'log_dir': './runs/superfloat_transformer'
    }

    # Start training
    model = train_model(config)

# Entry point to run locally
@app.local_entrypoint()
def main():
    train_and_upload.remote()