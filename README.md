---

# **SuperFloat: Accelerators for AI on Edge. Reimagined.**

This repository contains the code, methods, and scripts for implementing **Superfloat Quantization** and **Lottery Ticket Hypothesis (LTH)** techniques for optimizing neural networks. The repository focuses on various quantization algorithms, model evaluations, and fine-tuning techniques to minimize perplexity and stabilize activations.

---

## **What is Superfloat?**  

**Superfloat** is a custom quantization algorithm that operates with a **scalable precision format**. Unlike traditional floating-point systems (e.g., IEEE-754), Superfloat removes the mantissa entirely and focuses solely on the **exponent** for precision representation.  

### **Key Features**:  
1. **Sign-Exponent Representation**:  
   - Superfloat (SFx) uses `1 bit` for the **sign** and allocates the remaining `x-1 bits` for the **exponent**.  
   - For instance, in **SF16**:  
     - 1 bit → Sign  
     - 15 bits → Exponent  

2. **Clamping Range**:  
   - All values are clamped within the range `[-1, 1]`. This ensures activation and parameter stability, reducing the likelihood of exploding or vanishing gradients.

3. **Bit-width Flexibility**:  
   - Superfloat supports variable precision formats, scaling between **3-bit and 16-bit**:  
     - Lower precision (e.g., **SF4**) → Faster computation, reduced model size.  
     - Higher precision (e.g., **SF16**) → Improved accuracy while maintaining efficient quantization.

4. **Gradient and Activation Capping**:  
   - To stabilize the training process, gradients and activations are **capped** at -1 and +1.

### **Advantages of Superfloat**:  
- Saves **precision** without a significant drop in accuracy.  
- Reduces **computational complexity** compared to traditional floating-point representations.  
- Allows adaptive scaling for diverse quantization requirements.

---

**Conversion FP32 - SF(4-16)**

a standard 32-bit floating-point number is converted into a custom superfloat representation with a variable-sized mantissa.

- **Clamp Input Range** – The input value is restricted to the range (-1, 1). If the value exceeds this, it is set to a predefined maximum value.
    
- **Extract Sign Bit** – The sign bit is determined and stored separately, while the value is converted to its absolute form.
    
- **Compute Mantissa** – The fractional value is scaled by `2^mantissa_bits` to convert it into an integer representation.
    
- **Bit Packing** – The sign bit and mantissa are arranged into a custom format, with the mantissa shifted to fit within a float-sized bit structure.
    
- **Bitwise Reinterpretation** – The constructed bit pattern is reinterpreted as a floating-point number and returned.

---
## **What is WASQ?**  

**WASQ** stands for **Weight and Activation Superfloat Quantization**. It is a **hybrid quantization framework** that leverages Superfloat precision to optimize both model weights and activations.

### **Key Characteristics of WASQ**:  
1. **Weight Quantization**:  
   - Model weights are converted to **Superfloat precision** (SFx) without requiring complex computations like mantissa adjustments.  

2. **Activation Quantization**:  
   - Activations are clamped and quantized within a stable range to prevent issues such as exploding activations.

3. **Optimization Algorithms**:  
   - WASQ includes customized algorithms like **WASQ OPT** and **Full Parameter Method (FPM)** to balance accuracy and convergence speed.

4. **Scalability**:  
   - WASQ supports **multi-bit quantization** (from 4-bit to 16-bit), making it adaptable for different deployment environments, such as:  
     - **Edge devices** → Lower precision for speed and memory savings.  
     - **Servers** → Higher precision for accuracy-sensitive tasks.

### **WASQ + Lottery Ticket Hypothesis (LTH)**  
WASQ integrates **LTH** to identify specific weights that are critical for maintaining model performance after quantization. By fine-tuning only the **essential weights**, WASQ reduces computational overhead while achieving high accuracy.

---

## **Files Overview**

1. **[Quant_Dequant.ipynb](Quant_Dequant.ipynb)**  
   Contains the implementation of basic Superfloat quantization and dequantization functions.

2. **[sf16quant.ipynb](sf16quant.ipynb)**  
   Builds on Superfloat quantization functions, specifically for **SF16 precision**.

3. **[lth_analysis.py](lth_analysis.py)**  
   Analyzes **activation magnitude distribution** for **LTH**. It compares activation patterns of original and quantized models.

4. **[lth_trainer.py](lth_trainer.py)**  
   The **LTH trainer** script for fine-tuning models based on the Lottery Ticket Hypothesis technique.

5. **[wasq_eval.py](wasq_eval.py)**  
   Calculates **perplexity** for a series of models, grouped by context length, epochs, or model species.

6. **[wasq_inference.py](wasq_inference.py)**  
   Provides inference capabilities for **individual** or **multiple WASQ-quantized models**.

7. **[wasq_fasteropt.py](wasq_fasteropt.py)**  
   An optimized version of the **OPT algorithm** implemented in `wasq_opt.py`.

8. **[wasq_opt.py](wasq_opt.py)**  
   Core implementation of the WASQ OPT algorithm.

9. **[wasq_fpm.py](wasq_fpm.py)**  
   Implements the **Full Parameter Method** (FPM) for WASQ quantization.

10. **[wasq_vanilla.py](wasq_vanilla.py)**  
    Baseline implementation of the **Vanilla algorithm** for WASQ.

11. **[results](results/)**  
    Contains outputs of model tests, perplexity scores, and supplementary studies.

---

## **Scaling Laws**

### 1. **Maximum Context Length Barrier - Perplexity Factor**  
For a model with `n` parameters, a calibration dataset of maximum input length `c`, **three-shot quantization fine-tuning**, and Superfloat precision bit `x` (where `4 ≤ x ≤ 16`):  

\[
P = f(n, c, 3, x)
\]

- **Lower P** indicates better model understanding and calibration performance.

---

### 2. **Maximum Neuron Spread Factor**  
This scaling law uses the **Lottery Ticket Hypothesis** for WASQ quantization to stabilize activations:

1. Perform a forward pass using the **original model** and record the average magnitudes of activations across all layers.  
2. Perform the same for the **vanilla quantized model** to observe how quantization impacts activation magnitudes.  
3. Rank layers based on the **difference in activation magnitudes** between the original and quantized models.  
4. Identify and **cluster layers** with significant deviations to address issues like exploding/vanishing activations.  
5. Fine-tune or analyze these clusters to ensure stable activations and minimal performance degradation.

The law establishes that the **maximum neuron spread** (region targeted for fine-tuning/updating) is a function of:  
- **Activation magnitude**  
- **Activation fracture** (spread of how a weight affects neighboring weights during backpropagation)

---

## **Quantization Algorithms**

The repository explores three quantization approaches:

1. **Superfloat Precision**: Custom precision without mantissa, clamped within `[-1, 1]` for stability.  
2. **WASQ OPT**: Optimized quantization with faster convergence.  
3. **Full Parameter Method (FPM)**: Retrains all parameters for higher accuracy.

---

## **Usage**

### **Setup**  
Clone the repository and install dependencies:

```bash
git clone https://github.com/aloshdenny/superfloat
cd superfloat
pip install -r requirements.txt
```

### **Running Scripts**  

- Train with **LTH**:  
   ```bash
   python lth_trainer.py
   ```

- Evaluate Perplexity:  
   ```bash
   python wasq_eval.py
   ```

- Perform Inference:  
   ```bash
   python wasq_inference.py
   ```

---

## **Results**

The results folder contains:  
- **Perplexity scores** for different model configurations.  
- **Activation magnitude comparisons** before and after quantization.  
- Supplementary studies showcasing model performance.

---

## **Chip-1: Atreides**

Atreides is an ASIC accelerator designed specifically for Superfloat-based inference. We redesigned the systolic array to support SFx operations, adopting a modded RV32 ISA and faster Fused-Multiply-Adder (FMA) units. The end goal is not convention—it's breaking the rules of computing and physics to achieve faster inference, lower memory consumption, and the same accuracy.

## FMA in Atreides

Below is an image showing the FMA in Atreides:

![FMA](results/FMA.png)

## Expanded View of Chip-1's Architecture

An expanded view of Chip-1's architecture includes non-unified memory blocks (subject to unification), cache, control store (modded RV32 ISA), and an array of FMAs:

![Chip-1 Architecture](results/hardware%20architecture.png)

## FPGA Floorplan

The current floorplan for the FPGA design is shown below:

![FPGA Floorplan](results/floorplan.png)

### FPGA Functional Units Design

#### 1. 8 x 16-bit Shift Register (simplified)

![FPGA Floorplan](results/shift_register.png)

#### 2. Activation Unit (simplified)

![FPGA Floorplan](results/activation_unit.png)

#### 3. Cycle Count Logic

![FPGA Floorplan](results/cycle_count_logic.png)

## Instruction Set

The current instruction set for the FPGA architecture is show below:

| Instruction | Opcode(4) | Op 1(4) | Op 2(4) | Op 3(4) | Description                                                                           |
|-------------|-----------|---------|---------|---------|---------------------------------------------------------------------------------------|
| STR         | 0001      | addr    | row     | col     | Stores the matrix data from activation unit buffer into specified address in memory   |
| LDR         | 0010      | addr    | row     | col     | Loads the matrix at addr into the Row Shift Buffer                                    |
| LDC         | 0011      | addr    | row     | col     | Loads the matrix at addr into the Column Shift Buffer                                 |
| MATMUL      | 0100      | -       | -       | -       | Performs matrix multiplication using data in Row Shift Buffer and Column Shift Buffer |
| RELU        | 0101      | -       | -       | -       | Performs ReLU activation function on Systolic Array output                            |
| LIN         | 0110      | -       | -       | -       | Performs Linear activation function on Systolic Array output                          |
| NOP         | 0000      | -       | -       | -       | No Operation                                                                          |


### FPGA floorplan (ISA integrated)

The FPGA floorplan integrated with instruction set is shown below:


![FPGA Floorplan](results/isa_integrated_floorplan.png)

## **Contributions**

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## **Sponsors**

We would like to thank our sponsors for their support:

<div style="display: flex; justify-content: space-between;">
  <img src="https://pbs.twimg.com/profile_images/1848649662825406464/NFqR2OSK_400x400.jpg" width="200"/>
  <img src="https://media.licdn.com/dms/image/v2/D4E0BAQHKQl06Q7cd0A/company-logo_200_200/company-logo_200_200/0/1696367982134/modal_labs_logo?e=1743033600&v=beta&t=nMupgj5Hu8hl1mHyp0pDBJbqOEbGkRLmy7TZOTMuEZM" width="200"/>
  <img src="https://pbs.twimg.com/profile_images/1247800867777994755/JjEBNHba_400x400.jpg" width="200"/>
  <img src="https://styles.redditmedia.com/t5_bxucfi/styles/profileIcon_xjiodpqbkvbd1.jpg?width=256&height=256&frame=1&auto=webp&crop=256:256,smart&s=bda66bfc6dae1682cf1e5351a48ae8e473e12203" width="200">
</div>

---

## **License**

This project is licensed under the MIT License.

---
