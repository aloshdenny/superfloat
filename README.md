# **Superfloat: Accelerators for AI on Edge. Reimagined.**

This repository contains the code, methods, and scripts for implementing **Superfloat Quantization** and **Lottery Ticket Hypothesis (LTH)** techniques for optimizing neural networks. The repository focuses on various quantization algorithms, model evaluations, and fine-tuning techniques to minimize perplexity and stabilize activations.

---

## **What is Superfloat?**  

**Superfloat** is a custom quantization algorithm that operates with a **scalable precision format**. Unlike traditional floating-point systems (IEEE-754), Superfloat removes the mantissa entirely and focuses solely on the **exponent** for precision representation.  

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

A standard 32-bit floating-point number is converted into a custom superfloat representation with a variable-sized mantissa.

- **Clamp Input Range** – The input value is restricted to the range (-1, 1). If the value exceeds this, it is set to a predefined maximum value.
    
- **Extract Sign Bit** – The sign bit is determined and stored separately, while the value is converted to its absolute form.
    
- **Compute Mantissa** – The fractional value is scaled by `2^mantissa_bits` to convert it into an integer representation.
    
- **Bit Packing** – The sign bit and mantissa are arranged into a custom format, with the mantissa shifted to fit within a float-sized bit structure.
    
- **Bitwise Reinterpretation** – The constructed bit pattern is reinterpreted as a floating-point number and returned.

---