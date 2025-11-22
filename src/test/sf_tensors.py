import os
import re
import numpy as np

# =============================
# Utilities for formatting/IO
# =============================

def format_matrix(matrix):
    rows = []
    for row in matrix:
        row_str = ', '.join(f'{x:6.2f}' for x in row)
        rows.append(f'    [{row_str}]')
    return '[\n' + ',\n'.join(rows) + '\n]'


def save_tensor_to_txt(tensor, filename):
    """Saves a 2D tensor (list of lists) to a .txt file in formatted rows.
    Wraps the entire tensor in square brackets and ensures a trailing comma at end of every line.
    """
    with open(filename, "w") as f:
        if not tensor:
            f.write("[]\n")
            print(f"✅ Saved: {filename}")
            return

        f.write("[\n")
        for row in tensor:
            row_str = "  [" + ", ".join(str(val) for val in row) + "],\n"
            f.write(row_str)
        f.write("]\n")
    print(f"✅ Saved: {filename}")


# =============================
# Q1.15 (1 sign + 15 fractional) helpers
# Sign-magnitude convention used in the original code.
# =============================

def decimal_to_sf(decimal_val: float, n_bits: int = 15) -> str:
    """
    Converts a decimal number to signed fixed-point binary format with overflow clamping.
    Range is [-1.0 + 2^-n, 1.0 - 2^-n]. Uses sign-magnitude textual format: 's.ffff...'.
    """
    max_val = 1.0 - (2 ** (-n_bits))
    min_val = -1.0 + (2 ** (-n_bits))

    if decimal_val >= 1.0:
        decimal_val = max_val
    elif decimal_val < -1.0:
        decimal_val = min_val

    sign_bit = '0' if decimal_val >= 0 else '1'
    abs_val = abs(decimal_val)

    frac_binary = ""
    for _ in range(n_bits):
        abs_val *= 2
        if abs_val >= 1:
            frac_binary += '1'
            abs_val -= 1
        else:
            frac_binary += '0'

    return f"{sign_bit}.{frac_binary}"


def sf_to_decimal(sf_binary: str) -> float:
    """Converts signed fixed-point text 's.ffff' (sign-magnitude) to decimal in [-1, 1)."""
    sign_bit = sf_binary[0]
    frac_part = sf_binary[2:]

    decimal_val = 0.0
    for i, bit in enumerate(frac_part):
        if bit == '1':
            decimal_val += 2 ** (-(i + 1))
    if sign_bit == '1':
        decimal_val = -decimal_val
    return decimal_val


def sf_binary_to_hex(sf_binary: str, n_bits: int = None) -> str:
    """
    Converts signed fixed-point binary string 's.ffff' (sign-magnitude) to packed hex of (1+n_bits) bits.
    """
    if n_bits is None:
        n_bits = len(sf_binary) - 2
    sign_bit = sf_binary[0]
    frac_part = sf_binary[2:].ljust(n_bits, '0')[:n_bits]
    full_binary = sign_bit + frac_part
    binary_int = int(full_binary, 2)
    total_bits = n_bits + 1
    hex_digits = (total_bits + 3) // 4
    return f"0x{binary_int:0{hex_digits}X}"


def tensor_to_hex(tensor: list, n_bits: int = 15) -> list:
    hex_tensor = []
    for row in tensor:
        hex_row = []
        for val in row:
            if isinstance(val, str):
                hex_row.append(sf_binary_to_hex(val, n_bits))
            else:
                sf_val = decimal_to_sf(val, n_bits)
                hex_row.append(sf_binary_to_hex(sf_val, n_bits))
        hex_tensor.append(hex_row)
    return hex_tensor


def save_hex_tensor(tensor, filename, n_bits=15):
    if not tensor:
        save_tensor_to_txt([], filename)
        return
    if isinstance(tensor[0][0], str) and tensor[0][0].startswith(('0.', '1.')):
        hex_tensor = tensor_to_hex(tensor, n_bits)
    else:
        sf_tensor = [[decimal_to_sf(tensor[i][j], n_bits) for j in range(len(tensor[i]))] for i in range(len(tensor))]
        hex_tensor = tensor_to_hex(sf_tensor, n_bits)
    save_tensor_to_txt(hex_tensor, filename)


# =============================
# Pretty printers
# =============================

def print_tensor(tensor, title, precision=6):
    print(f"{title}:")
    for row in tensor:
        formatted_row = []
        for val in row:
            if isinstance(val, str):
                formatted_row.append(f"{val:>12}")
            else:
                formatted_row.append(f"{val:{precision+6}.{precision}f}")
        print("  [" + ", ".join(formatted_row) + "]")
    print()


def print_tensor_hex(tensor, title, n_bits: int = 15):
    print(f"{title} (Hexadecimal):")
    if tensor and isinstance(tensor[0][0], str) and tensor[0][0].startswith(('0.', '1.')):
        hex_tensor = tensor_to_hex(tensor, n_bits)
    elif tensor and isinstance(tensor[0][0], (int, float)):
        sf_tensor = [[decimal_to_sf(tensor[i][j], n_bits) for j in range(len(tensor[i]))] for i in range(len(tensor))]
        hex_tensor = tensor_to_hex(sf_tensor, n_bits)
    else:
        hex_tensor = tensor
    for row in hex_tensor:
        formatted_row = [f"{val:>8}" for val in row]
        print("  [" + ", ".join(formatted_row) + "]")
    print()


def view_tensors_comparison(tensor_a, tensor_b, sf_result, n_bits: int = 15):
    print("\n" + "="*80)
    print("TENSOR COMPARISON: BINARY, DECIMAL, AND HEXADECIMAL VIEWS")
    print("="*80)

    a_sf = [[decimal_to_sf(tensor_a[i][j], n_bits) for j in range(len(tensor_a[i]))] for i in range(len(tensor_a))]
    b_sf = [[decimal_to_sf(tensor_b[i][j], n_bits) for j in range(len(tensor_b[i]))] for i in range(len(tensor_b))]

    print(f"\nTENSOR A:")
    print("-" * 40)
    print_tensor(tensor_a, "Decimal", 3)
    print_tensor(a_sf, "SF Binary")
    print_tensor_hex(a_sf, "Hexadecimal", n_bits)

    print(f"\nTENSOR B:")
    print("-" * 40)
    print_tensor(tensor_b, "Decimal", 3)
    print_tensor(b_sf, "SF Binary")
    print_tensor_hex(b_sf, "Hexadecimal", n_bits)

    print(f"\nSF MULTIPLICATION RESULT:")
    print("-" * 40)
    decimal_result = [[sf_to_decimal(sf_result[i][j]) for j in range(len(sf_result[i]))] for i in range(len(sf_result))]
    print_tensor(decimal_result, "Decimal", 6)
    print_tensor(sf_result, "SF Binary")
    print_tensor_hex(sf_result, "Hexadecimal", n_bits)

    print(f"Format Details:")
    print(f"  Bits: 1 sign + {n_bits} fractional = {n_bits + 1} total bits")
    print(f"  Hex digits: {(n_bits + 1 + 3) // 4}")
    print(f"  Range: [{-1.0 + (2**(-n_bits)):.10f}, {1.0 - (2**(-n_bits)):.10f}]")
    print(f"  Resolution: {2**(-n_bits):.10f}")
    print("="*80)


# ==============================================
# NEW: Integer-path helpers for guarded accumulation
# ==============================================

def clamp_q_int(x: int, n_bits: int) -> int:
    """Clamp an integer Q0.n_bits (sign-magnitude range) to representable range.
    Max:  +(2^n_bits - 1);  Min: -(2^n_bits - 1)
    (We avoid -2^n_bits to stay consistent with sign-magnitude textual format.)
    """
    qmax = (1 << n_bits) - 1
    qmin = -qmax
    return max(qmin, min(qmax, x))


def decimal_to_qint(x: float, n_bits: int) -> int:
    """Quantize decimal x in [-1,1) to signed integer with n_bits fractional bits.
    Uses rounding-to-nearest (ties to +inf for simplicity). Clamped to sign-magnitude range.
    """
    scale = 1 << n_bits
    # Clamp to [-1 + ulp, 1 - ulp]
    x = max(-1.0 + 1.0/scale, min(1.0 - 1.0/scale, x))
    q = int(np.round(x * scale))
    # Ensure sign-magnitude-compliant bounds
    return clamp_q_int(q, n_bits)


def qint_to_sf_string(q: int, n_bits: int) -> str:
    sign_bit = '0' if q >= 0 else '1'
    mag = abs(q)
    frac_binary = format(mag, f'0{n_bits}b')
    return f"{sign_bit}.{frac_binary}"


def qint_to_decimal(q: int, n_bits: int) -> float:
    return q / float(1 << n_bits)


# =============================================================
# UPDATED: Matrix multiply with 32-bit guarded accumulation
# =============================================================

def sf_tensor_mul(A: list, B: list, n_bits: int = 15, acc_bits: int = 31) -> dict:
    """
    Multiply two matrices using Q1.n_bits arithmetic with **guarded accumulation**.

    Algorithm (integer path):
      1) Quantize inputs to Q integers (signed, n_bits fractional).
      2) For each output C[i,j], compute sum_k (A_q[i,k] * B_q[k,j]) in a wide accumulator (acc_bits).
      3) After the full dot-product, ROUND-to-nearest by adding +/- 2^(n_bits-1), then shift right by n_bits.
      4) Clamp to representable Q range (sign-magnitude compliant), convert to SF text, and decimal.

    Notes:
      - Products are kept at full precision until the *end* of the dot-product.
      - Accumulator width defaults to 31 bits (1 sign + 30 mantissa) to preserve full precision
        from Q1.15 × Q1.15 multiplication (15 + 15 = 30 fractional bits).
    """
    rows_a = len(A)
    cols_a = len(A[0]) if rows_a > 0 else 0
    rows_b = len(B)
    cols_b = len(B[0]) if rows_b > 0 else 0
    if cols_a != rows_b:
        raise ValueError(f"Cannot multiply matrices: {rows_a}x{cols_a} × {rows_b}x{cols_b}")

    # Quantize inputs to Q integers
    A_q = [[decimal_to_qint(A[i][k], n_bits) for k in range(cols_a)] for i in range(rows_a)]
    B_q = [[decimal_to_qint(B[k][j], n_bits) for j in range(cols_b)] for k in range(rows_b)]

    # Prepare outputs
    sf_result = [[None for _ in range(cols_b)] for _ in range(rows_a)]
    decimal_result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
    exact_result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
    clamped_exact_result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]

    total_error = 0.0
    max_error = 0.0
    input_overflow_count = 0  # not strictly needed now; inputs are quantized
    result_overflow_count = 0

    # Accumulator bounds for simulation (signed acc_bits)
    acc_max = (1 << (acc_bits - 1)) - 1
    acc_min = -(1 << (acc_bits - 1))

    scale = 1 << n_bits
    qmax = (1 << n_bits) - 1
    qmin = -qmax

    for i in range(rows_a):
        for j in range(cols_b):
            acc = 0  # wide accumulator
            exact_sum = 0.0

            for k in range(cols_a):
                prod = int(A_q[i][k]) * int(B_q[k][j])  # up to 2*n_bits fractional bits
                acc += prod
                # Simulate saturating accumulator to acc_bits if desired
                if acc > acc_max:
                    acc = acc_max
                elif acc < acc_min:
                    acc = acc_min

                exact_sum += A[i][k] * B[k][j]

            # TRUNCATE to return to Q1.n_bits (matching Verilog hardware behavior)
            # Verilog: output_result = {sign_out, mult_result[29:15]};
            # This is simple right-shift without rounding
            q_out = acc >> n_bits  # back to Q1.n_bits integer via truncation

            # Clamp to representable sign-magnitude range
            if q_out > qmax:
                q_out = qmax; result_overflow_count += 1
            elif q_out < qmin:
                q_out = qmin; result_overflow_count += 1

            # Store outputs
            sf_str = qint_to_sf_string(q_out, n_bits)
            sf_result[i][j] = sf_str
            dec_val = qint_to_decimal(q_out, n_bits)
            decimal_result[i][j] = dec_val
            exact_result[i][j] = exact_sum

            # For fair error, compare to exact but clamped to representable range
            exact_clamped = max(-1.0 + 1.0/scale, min(1.0 - 1.0/scale, exact_sum))
            clamped_exact_result[i][j] = exact_clamped

            err = abs(exact_clamped - dec_val)
            total_error += err
            if err > max_error:
                max_error = err

    return {
        'sf_result': sf_result,
        'decimal_result': decimal_result,
        'exact_result': exact_result,
        'clamped_exact_result': clamped_exact_result,
        'total_error': total_error,
        'average_error': total_error / (rows_a * cols_b),
        'max_error': max_error,
        'input_overflow_count': input_overflow_count,
        'result_overflow_count': result_overflow_count,
        'result_shape': (rows_a, cols_b)
    }


# =============================
# Example driver (same UX as before)
# =============================

def load_matrix(filename):
    mat = []
    with open(filename, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('[') and line.endswith(']'):
                line = line[1:-1].strip()
            if line.endswith(','):
                line = line[:-1].strip()
            if ',' in line:
                parts = [p.strip() for p in line.split(',') if p.strip() != '']
            else:
                parts = line.split()
            if not parts:
                parts = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
            try:
                row = [float(p) for p in parts]
            except ValueError:
                parts = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
                row = [float(p) for p in parts]
            if row:
                mat.append(row)
    return mat


def main():
    matrix_size = 8

    # If A_matrix.txt / B_matrix.txt don't exist, generate them
    # if not (os.path.exists(f"superfloat/src/test/A_{matrix_size}x{matrix_size}_matrix.txt") and os.path.exists(f"superfloat/src/test/B_{matrix_size}x{matrix_size}_matrix.txt")):
    #     A = np.round(np.random.uniform(-1, 1, (matrix_size, matrix_size)), 2)
    #     B = np.round(np.random.uniform(-1, 1, (matrix_size, matrix_size)), 2)
    #     with open(f"superfloat/src/test/A_{matrix_size}x{matrix_size}_matrix.txt", "w") as f:
    #         f.write(format_matrix(A))
    #     with open(f"superfloat/src/test/B_{matrix_size}x{matrix_size}_matrix.txt", "w") as f:
    #         f.write(format_matrix(B))

    A = load_matrix(f"superfloat/src/test/A_{matrix_size}x{matrix_size}_matrix.txt")
    B = load_matrix(f"superfloat/src/test/B_{matrix_size}x{matrix_size}_matrix.txt")

    if not A or not B:
        raise ValueError("One of the input files is empty or could not be parsed: A_matrix.txt, B_matrix.txt")
    if len(A[0]) != len(B):
        raise ValueError(f"Matrix dimension mismatch for multiplication: A is {len(A)}x{len(A[0])}, B is {len(B)}x{len(B[0]) if B and B[0] else 0}")

    # # Previews in SF binary for inputs
    A_sf = [[decimal_to_sf(A[i][j], 15) for j in range(len(A[i]))] for i in range(len(A))]
    B_sf = [[decimal_to_sf(B[i][j], 15) for j in range(len(B[i]))] for i in range(len(B))]

    # # Guarded accumulate multiply with 31-bit accumulator (1 sign + 30 mantissa)
    result = sf_tensor_mul(A, B, n_bits=15, acc_bits=31)

    # Save binary (SF) tensors
    save_tensor_to_txt(A_sf, f"superfloat/src/test/A_{matrix_size}x{matrix_size}_binary.txt")
    save_tensor_to_txt(B_sf, f"superfloat/src/test/B_{matrix_size}x{matrix_size}_binary.txt")
    save_tensor_to_txt(result['sf_result'], f"superfloat/src/test/Result_{matrix_size}x{matrix_size}_binary.txt")

    # Save hexadecimal tensors
    save_hex_tensor(A_sf, f"superfloat/src/test/A_{matrix_size}x{matrix_size}_hex.txt", n_bits=15)
    save_hex_tensor(B_sf, f"superfloat/src/test/B_{matrix_size}x{matrix_size}_hex.txt", n_bits=15)
    save_hex_tensor(result['sf_result'], f"superfloat/src/test/Result_{matrix_size}x{matrix_size}_hex.txt", n_bits=15)


if __name__ == "__main__":
    main()