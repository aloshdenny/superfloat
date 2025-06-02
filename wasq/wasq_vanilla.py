import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from transformers import AutoModelForCausalLM


class Superfloat:
    """Simple Superfloat quantizer with encode/decode utilities."""

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
        self.max_val = 1 - 2 ** -self.mantissa_bits
        self.float_type = self.CASTING_TABLE[bits]

    def encode(self, value: torch.Tensor):
        clipped = torch.clamp(value, min=-self.max_val, max=self.max_val)
        mantissa = (torch.abs(clipped) * (2 ** self.mantissa_bits - 1) / self.max_val).floor().to(torch.int32)
        sign = (clipped < 0).to(torch.int32)
        encoded = (mantissa | (sign << self.mantissa_bits)).to(torch.int32)
        out_of_range = (value.abs() > self.max_val)
        return encoded, out_of_range

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        mantissa = encoded & ((1 << self.mantissa_bits) - 1)
        sign = (encoded >> self.mantissa_bits) & 1
        decoded = (mantissa.to(self.float_type) / (2 ** self.mantissa_bits - 1)) * self.max_val
        return decoded * (2 * sign - 1)

    def tensor_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        enc, _ = self.encode(tensor)
        return self.decode(enc)


class SFQuant(Function):
    """Straight-through estimator for Superfloat quantization."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, sf: "Superfloat"):
        encoded, mask = sf.encode(input)
        ctx.save_for_backward(mask)
        ctx.sf = sf
        return sf.decode(encoded)

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        # Pass gradients only where values were in range
        return grad_output * mask.to(grad_output.dtype), None


class QuantizedLinear(nn.Linear):
    """Linear layer with on-the-fly Superfloat decode and optional LSQ+ scale."""

    def __init__(self, in_features, out_features, sf: Superfloat, bias=True, k_outlier=0.005):
        super().__init__(in_features, out_features, bias)
        self.sf = sf

        # Split outlier channels that would overflow after quantisation
        with torch.no_grad():
            channel_max = self.weight.abs().max(dim=1).values
            k = max(1, int(k_outlier * out_features))
            self.outlier_idx = torch.topk(channel_max, k).indices
            mask = torch.ones(out_features, dtype=torch.bool)
            mask[self.outlier_idx] = False
            base_w = self.weight[mask].clone()
            self.register_buffer("encoded_weight", sf.encode(base_w)[0])
            self.register_parameter("scale", nn.Parameter(torch.ones(base_w.size(0))))
            self.register_parameter("outlier_weight", nn.Parameter(self.weight[self.outlier_idx].clone()))
            self.register_buffer("mask", mask)
        # Remove original parameter
        self.weight.requires_grad = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Decode base weight on the fly and apply LSQ+ scale
        decoded_base = self.sf.decode(self.encoded_weight) * self.scale.view(-1, 1)
        weight = self.weight.new_zeros(self.out_features, self.in_features)
        weight[self.mask] = decoded_base
        weight[self.outlier_idx] = self.outlier_weight
        return F.linear(input, weight, self.bias)


class ActivationQuant(nn.Module):
    """Module to quantise activations symmetrically with Superfloat."""

    def __init__(self, sf: Superfloat):
        super().__init__()
        self.sf = sf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SFQuant.apply(x, self.sf)


def compute_hessian_scores(model, data_loader, device, num_batches=1):
    """Approximate block-diagonal Hessian scores for parameters."""
    scores = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        logits = output.logits
        target = batch["input_ids"][:, 1:].contiguous()
        logits = logits[:, :-1].contiguous()
        loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
        grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], create_graph=False)
        for (name, _), g in zip([(n, p) for n, p in model.named_parameters() if p.requires_grad], grads):
            scores[name] += g.pow(2)
    return scores


def select_sf_bits(weight, score, bit_options=(16, 11, 8, 4), budget=1e-3):
    """Simple layer-adaptive bit-width search using a quantisation error budget."""
    for bits in sorted(bit_options, reverse=True):
        sf = Superfloat(bits)
        q = sf.tensor_quantize(weight)
        err = (weight - q).abs().mean() * score.mean()
        if err <= budget:
            return sf
    return Superfloat(bit_options[0])


def quantize_model(model, sf_options=(16, 11, 8, 4), data_loader=None, device="cpu"):
    """Quantise linear layers adaptively and insert activation quantisation."""
    if data_loader is not None:
        scores = compute_hessian_scores(model, data_loader, device)
    else:
        scores = {name: torch.ones_like(p) for name, p in model.named_parameters() if p.requires_grad}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            score = scores.get(f"{name}.weight", torch.ones_like(module.weight))
            sf = select_sf_bits(module.weight.data, score)
            qlinear = QuantizedLinear(module.in_features, module.out_features, sf, module.bias is not None)
            qlinear.bias = module.bias
            setattr(model, name.split(".")[-1], qlinear)
        elif isinstance(module, nn.Module) and not isinstance(module, ActivationQuant):
            module.register_forward_pre_hook(lambda m, inp: (SFQuant.apply(inp[0], Superfloat(11)),))
    return model


def main():
    model_name = "Qwen/Qwen2-0.5B"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Model loading may require network; placeholder path for offline usage
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./")
    model = model.to(device)

    # Dummy dataloader for Hessian approximation (replace with real data)
    dummy_input = torch.randint(0, 10, (1, 8))
    dummy_mask = torch.ones_like(dummy_input)
    dataset = [{"input_ids": dummy_input, "attention_mask": dummy_mask}]
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    print("Applying adaptive Superfloat quantisation...")
    quantized_model = quantize_model(model, data_loader=data_loader, device=device)

    save_path = "sf_vanilla_adaptive.pt"
    torch.save(quantized_model.state_dict(), save_path)
    print(f"Quantised model saved to {save_path}")


if __name__ == "__main__":
    main()
