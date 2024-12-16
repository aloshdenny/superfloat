Here's the **README** for your repository, with hyperlinks and explanations integrated for clarity.

---

# **WASQ Repository: Superfloat Quantization and LTH Implementation**

This repository contains the code, methods, and scripts for implementing **Superfloat Quantization** and **Lottery Ticket Hypothesis (LTH)** techniques for optimizing neural networks. The repository focuses on various quantization algorithms, model evaluations, and fine-tuning techniques to minimize perplexity and stabilize activations.

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

## **License**

This project is licensed under the MIT License.

---

Let me know if you'd like to tweak or add more sections! 🚀
