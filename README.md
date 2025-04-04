# SuperFloat: Accelerators for AI on Edge. Reimagined.

![SuperFloat Logo](https://via.placeholder.com/150x50) <!-- Add actual logo if available -->

This repository contains cutting-edge quantization techniques combining **Superfloat Quantization**, **Weight-aware Selective Quantization (WASQ)**, and **Simulated Annealing Multi-Prize Lottery Ticket Hypothesis (SA-MPLTH)** for optimizing neural networks on edge devices.

---

## Key Innovations

### 🚀 SuperFloat Quantization
A revolutionary numeric format that:
- Uses **sign-exponent only** representation (no mantissa)
- Operates within clamped `[-1, 1]` range for stability
- Supports **3-bit to 16-bit** precision
- Enables **74% memory reduction** in LLMs

### ⚡ WASQ Framework
Our Weight-aware Selective Quantization system featuring:
- 6 optimization algorithms (Vanilla to SA-MPLTH)
- Hardware-aware quantization
- RISC-V compatible instruction set
- **2.2× speedup** over conventional GPUs

### 🏆 SA-MPLTH Algorithm
Our novel three-pass quantization healing process:
1. **Subnetwork Prospecting**: Identifies robust quantized subnetworks
2. **Gradient-Guided Healing**: Fine-tunes with simulated annealing
3. **Ensemble Fusion**: Combines complementary subnetworks

---

## Chip-1: Atreides Accelerator

Our custom ASIC designed for Superfloat inference:

### Hardware Architecture
![Chip-1 Architecture](results/hardware%20architecture.png)

### Key Components
- **Fused Multiply-Add (FMA) Units**
  ![FMA Unit](results/FMA.png)
- **Non-Unified Memory Architecture**
- **Modified RV32 ISA** with custom instructions:
  
  | Instruction | Opcode | Description |
  |------------|--------|-------------|
  | MATMUL | 0100 | Matrix multiplication |
  | SFQUANT | 0111 | Superfloat quantization |
  | RELU | 0101 | Activation function |

### Performance
- **10.6 tokens/sec** on Raspberry Pi 5
- **74% memory reduction** for DeepSeek-R1
- **2.2× speedup** vs conventional GPUs

---

## Implementation Highlights

### Quantization Algorithms
| Algorithm | Use Case | Key Feature |
|-----------|----------|-------------|
| WASQ-Vanilla | Baseline | Simple SF8 quantization |
| WASQ-FPM | High accuracy | Full parameter retraining |
| SA-MPLTH | Optimal healing | Simulated annealing + LTH |

### Code Structure
```
superfloat/
├── core/                  # Core quantization algorithms
│   ├── superfloat.py      # Superfloat datatype implementation
│   ├── wasq_opt.py        # WASQ optimized quantizer
│   └── sa_mplth.py        # SA-MPLTH implementation
├── hardware/              # Atreides accelerator designs
│   ├── fma/               # Fused Multiply-Add units
│   └── isa/               # Modified RISC-V ISA
├── models/                # Quantized model zoo
├── scripts/               # Training/evaluation scripts
└── results/               # Benchmark results
```

---

## Getting Started

### Installation
```bash
git clone https://github.com/aloshdenny/superfloat-accelerator
cd superfloat-accelerator
pip install -r requirements.txt
```

### Basic Usage
```python
from core.superfloat import Superfloat
from core.sa_mplth import SA_MPLTH_Trainer

# Initialize 8-bit Superfloat quantizer
quantizer = Superfloat(bits=8)

# Load model and setup SA-MPLTH trainer
trainer = SA_MPLTH_Trainer(
    model=your_model,
    sf_quantizer=quantizer,
    config={
        'initial_temp': 1.0,
        'final_temp': 0.01,
        'pruning_rates': [0.7, 0.8, 0.9]
    }
)

# Run quantization-aware training
optimized_model = trainer.train()
```
---

## Performance Results

### Quantization Comparison
| Model | FP32 Size | SF8 Size | Perplexity Δ | Speedup |
|-------|-----------|----------|--------------|---------|
| DeepSeek-R1 | 28GB | 7.4GB | +0.9 | 10.6x |
| Llama 3.2 | 32GB | 9.3GB | +1.1 | 6.95x |
| Gemma-3 | 24GB | 10.1GB | +0.8 | 10.4x |

### SA-MPLTH Effectiveness
![Perplexity Improvement](results/perplexity_improvement.png) <!-- Add actual graph -->

---

## Roadmap
- [ ] Dynamic bit-width adjustment
- [ ] Neuromorphic integration
- [ ] Community-driven SF type extensions
- [ ] Atreides tapeout (2025 Q2)

---

## Sponsors & Acknowledgments
<div style="display: flex; justify-content: space-between;">
  <img src="assets/sponsor1.png" width="150"/>
  <img src="assets/sponsor2.png" width="150"/>
  <img src="assets/sponsor3.png" width="150"/>
</div>

Special thanks to Dr. Pramod Pavithran and Cochin University of Science and Technology for research support.

---

## License
MIT License © 2025 Alosh Denny @ EmelinLabs
```
