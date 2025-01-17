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
    )
    .apt_install("gcc", "python3-dev")  # Add necessary system libraries if needed
)

app = modal.App("qwen-sf4-experimental")

# Define the function that runs the script
@app.function(gpu="H100", image=image, timeout=86400)
def train_and_upload():
    import torch
    import gc
    import os
    import re
    import requests
    from tqdm import tqdm
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pandas as pd
    import math

    # Function to calculate perplexity
    def calculate_perplexity(model, dataloader, loss_fn, device):
        model.eval()  # Set model to evaluation mode
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                target = input_ids[:, 1:].contiguous()  # Shift targets by one token
                logits = logits[:, :-1].contiguous()  # Align logits with target

                loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
                total_loss += loss.item()
                total_steps += 1

        avg_loss = total_loss / total_steps
        perplexity = math.exp(avg_loss)  # Perplexity is the exponential of the average loss
        return perplexity

    # List of dataset URLs
    urls = [
        "https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated/resolve/main/data/train-00000-of-01650-f70471ee3deb09c0.parquet",
    ]

    # Local final output file path
    final_file_name = "train.parquet"

    # Check if the final file already exists
    if not os.path.exists(final_file_name):
        print(f"Downloading and combining dataset from {len(urls)} files...")

        # List to hold all the dataframes
        combined_df = pd.DataFrame()

        # Loop through each URL to download and combine the files
        for i, url in enumerate(urls):
            downloaded_file = f"temp_file_{i}.parquet"
            
            # Download the dataset
            print(f"Downloading dataset from {url}...")
            response = requests.get(url, stream=True)
            with open(downloaded_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded to {downloaded_file}.")

            # Read the downloaded parquet file and append to the combined dataframe
            df = pd.read_parquet(downloaded_file)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

            # Optionally remove the temporary file after reading
            os.remove(downloaded_file)

        # Save the combined dataframe as a final parquet file
        combined_df.to_parquet(final_file_name)
        print(f"Combined data saved to {final_file_name}.")
    else:
        print(f"{final_file_name} already exists. Skipping download.")


    # max_lengths = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    max_lengths = [2]
    bit = 4

    class Superfloat:
        # Mapping of bit-widths to floating-point types
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
            self.max_val = 1.0  # Default max value
            self.scale_factor = 1.0  # Initialize   scale factor
            self.float_type = self.CASTING_TABLE[bits]  # Set float_type based on bits

        def set_scale(self, weights, dim=None, percentile=None):
            if dim is not None:
                self.scale_factor = torch.max(torch.abs(weights), dim=dim, keepdim=True)[0]
            elif percentile:
                scale = torch.kthvalue(torch.abs(weights).view(-1), int(weights.numel() * percentile / 100))[0]
                self.scale_factor = scale
            else:
                self.scale_factor = torch.max(torch.abs(weights))

        def encode(self, value: torch.Tensor):
            scaled_value = value / self.scale_factor
            quantized_value = torch.round(scaled_value * (2**self.mantissa_bits - 1)) / (2**self.mantissa_bits - 1)
            return quantized_value.to(self.float_type)  # Cast to float_type

        def decode(self, quantized_value: torch.Tensor):
            decoded_value = quantized_value * self.scale_factor
            return decoded_value.to(self.float_type)  # Cast to float_type
        
    sf = Superfloat(bit)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model and tokenizer
    model_name = "Qwen/Qwen2-0.5B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./", token='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')
    tokenizer.pad_token = tokenizer.eos_token

    def quantize_model(model, sf_type, dim=None, percentile=None):
        for name, param in model.named_parameters():
            # Cast weights to the correct float_type before quantization
            param.data = param.data.to(sf_type.float_type)
            
            # Set scale and quantize
            sf_type.set_scale(param.data, dim=dim, percentile=percentile)
            quantized_param = sf_type.encode(param.data)
            param.data = quantized_param.data
        return model
    
    def load_checkpoint(model, sf_bits, suffix="opt", device="cuda"):
        """
        Load the latest checkpoint based on the provided Superfloat bitwidth and filename suffix.

        Args:
            quantized_model: The model to load the checkpoint into.
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
            return quantize_model(model, sf), 0

        # Extract epoch numbers and sort by latest epoch
        epochs_and_files = [
            (int(checkpoint_pattern.match(f).group(1)), f) for f in checkpoint_files
        ]
        latest_epoch, latest_checkpoint = max(epochs_and_files, key=lambda x: x[0])

        # Load the latest checkpoint
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)

        return model, latest_epoch
    
    # Pre-training parameter check to ensure they are within range
    def check_parameters_in_range(model, sf):
        out_of_range_params = []
        for name, param in model.named_parameters():
            if not torch.all(torch.abs(param.data) <= sf.max_val):
                out_of_range_params.append(name)
        if out_of_range_params:
            print(f"Warning: The following parameters are out of range:")
            for param_name in out_of_range_params:
                print(f"- {param_name}")
        else:
            print("All parameters are within the valid range.")


    def prepare_dataset(tokenizer, max_length=1):
        dataset = Dataset.from_parquet("train.parquet")

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=dataset.column_names
        )
        return tokenized_dataset

    def collate_fn(batch):
        input_ids = torch.stack(
            [torch.tensor(example["input_ids"]) for example in batch]
        )
        attention_mask = torch.stack(
            [torch.tensor(example["attention_mask"]) for example in batch]
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}


    # Loop over different max_length values
    for max_length in max_lengths:
        print(f"Starting training for max_length = {max_length}")

        # Prepare Dataset
        tokenized_dataset = prepare_dataset(tokenizer, max_length=max_length)
        train_dataloader = DataLoader(
            tokenized_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./", token='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')
        model = model.to(sf.float_type).to(device)

        quantized_model, last_epoch = load_checkpoint(model, sf.bits, suffix="opt", device=device)
        quantized_model.to(device)
        print(f"Resuming training from epoch {last_epoch + 1}.")

        # Check if model parameters are within range before training
        check_parameters_in_range(quantized_model, sf)

        # del model
        torch.cuda.empty_cache()
        gc.collect()

        optimizer = torch.optim.Adam(quantized_model.parameters(), lr=1e-5, eps=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Calculate and print the original model's perplexity before training
        print("Calculating original model perplexity...")
        original_perplexity = calculate_perplexity(model, train_dataloader, loss_fn, device)
        print(f"Original model perplexity: {original_perplexity:.4f}")

        num_epochs = 10
        accumulation_steps = 16

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_iterator = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Epoch {epoch + 1}/{num_epochs}",
            )

            for step, batch in epoch_iterator:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = quantized_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                target = input_ids[:, 1:].contiguous()
                logits = logits[:, :-1].contiguous()

                loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
                loss = loss / accumulation_steps
                loss.backward()

                epoch_loss += loss.item() * accumulation_steps

                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_iterator.set_postfix({"Loss": f"{loss.item() * accumulation_steps:.4f}"})

            epoch_loss /= len(train_dataloader)
            print(f"Epoch {epoch + 1} completed with average loss: {epoch_loss:.4f}")

            # Calculate and print the perplexity after each epoch
            epoch_perplexity = calculate_perplexity(quantized_model, train_dataloader, loss_fn, device)
            print(f"Epoch {epoch + 1} perplexity: {epoch_perplexity:.4f}")

            model_path = f"sf{sf.bits}_{max_length}_{epoch + 1}_opt"
            torch.save(quantized_model.state_dict(), model_path)

            # Upload model to Hugging Face
            os.system(
                f"huggingface-cli upload aoxo/qwen2-idkwtf {model_path} --token='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'"
            )

        del quantized_model
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Completed training for max_length = {max_length}")

# Entry point to run locally
@app.local_entrypoint()
def main():
    train_and_upload.remote()