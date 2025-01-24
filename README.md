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
git clone <repo-url>
cd <repo-folder>
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

## **Contributions**

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## **Sponsors**

We would like to thank our sponsors for their support:

### E2E Cloud
<img src="https://pbs.twimg.com/profile_images/1848649662825406464/NFqR2OSK_400x400.jpg" width="200"/>

### Modal
<img src="https://media.licdn.com/dms/image/v2/D4E0BAQHKQl06Q7cd0A/company-logo_200_200/company-logo_200_200/0/1696367982134/modal_labs_logo?e=1743033600&v=beta&t=nMupgj5Hu8hl1mHyp0pDBJbqOEbGkRLmy7TZOTMuEZM" width="200"/>

### Tencent
<img src="https://pbs.twimg.com/profile_images/1247800867777994755/JjEBNHba_400x400.jpg" width="200"/>

### dat1.co
<img src="https://dat1.co/hs-fs/hubfs/photo_2024-06-14_19-48-12-1.jpg?width=114&height=75&name=photo_2024-06-14_19-48-12-1.jpg" width="200">

---

## **License**

This project is licensed under the MIT License.

---
