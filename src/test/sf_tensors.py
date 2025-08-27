def sf_mul(bin1: str, bin2: str, n_bits: int = None) -> str:
    """
    Multiplies two signed fixed-point binary fractional numbers in s.xxx format.
    
    Args:
    - bin1, bin2: binary strings like '0.101' or '1.011'
    - n_bits: number of fractional bits (excluding sign). If None, inferred.
    
    Returns:
    - Result in s.xxx format with sign bit and n_bits fractional bits.
    """
    # Extract sign and fractional parts
    sign1, frac1 = bin1[0], bin1[2:]
    sign2, frac2 = bin2[0], bin2[2:]
    # Infer n_bits if not given
    if n_bits is None:
        n_bits = max(len(frac1), len(frac2))
    # Pad fractions to match n_bits
    frac1 = frac1.ljust(n_bits, '0')
    frac2 = frac2.ljust(n_bits, '0')
    # Convert to integers
    int1 = int(frac1, 2)
    int2 = int(frac2, 2)
    # Apply signs
    if sign1 == '1':
        int1 = -int1
    if sign2 == '1':
        int2 = -int2
    # Multiply
    product = int1 * int2
    # Result needs 2 * n_bits for full precision
    product_bits = 2 * n_bits
    abs_product = abs(product)
    product_bin = bin(abs_product)[2:].zfill(product_bits)
    # Take the top n_bits as fractional result
    fractional_part = product_bin[:n_bits]
    # Determine sign bit
    sign_bit = '0' if product >= 0 else '1'
    return f"{sign_bit}.{fractional_part}"


def decimal_to_sf(decimal_val: float, n_bits: int = 8) -> str:
    """
    Converts a decimal number to signed fixed-point binary format with overflow clamping.
    
    Args:
    - decimal_val: decimal number (will be clamped to [-1, 1) range)
    - n_bits: number of fractional bits
    
    Returns:
    - Binary string in s.xxx format
    """
    # Clamp to valid SF range
    max_val = 1.0 - (2 ** (-n_bits))  # Maximum positive value (just under 1.0)
    min_val = -1.0  # Minimum negative value
    
    if decimal_val >= 1.0:
        decimal_val = max_val
    elif decimal_val < -1.0:
        decimal_val = min_val
    
    # Determine sign
    sign_bit = '0' if decimal_val >= 0 else '1'
    abs_val = abs(decimal_val)
    
    # Convert fractional part to binary
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
    """
    Converts signed fixed-point binary format to decimal.
    
    Args:
    - sf_binary: binary string in s.xxx format
    
    Returns:
    - Decimal equivalent
    """
    sign_bit = sf_binary[0]
    frac_part = sf_binary[2:]
    
    # Convert fractional part to decimal
    decimal_val = 0
    for i, bit in enumerate(frac_part):
        if bit == '1':
            decimal_val += 2 ** (-(i + 1))
    
    # Apply sign
    if sign_bit == '1':
        decimal_val = -decimal_val
    
    return decimal_val


def sf_mul_dec(dec1: float, dec2: float, n_bits: int = 8) -> dict:
    """
    Multiplies two decimal numbers using SF arithmetic with overflow clamping.
    
    Args:
    - dec1, dec2: decimal numbers (will be clamped to SF range if needed)
    - n_bits: number of fractional bits for binary representation
    
    Returns:
    - Dictionary with 'sf_binary', 'decimal', 'inputs_sf', 'exact_decimal', and overflow info
    """
    # Store original values for exact calculation
    original_dec1, original_dec2 = dec1, dec2
    
    # Clamp inputs to valid SF range
    max_val = 1.0 - (2 ** (-n_bits))
    min_val = -1.0
    
    input1_overflow = False
    input2_overflow = False
    
    if dec1 >= 1.0:
        dec1 = max_val
        input1_overflow = True
    elif dec1 < -1.0:
        dec1 = min_val
        input1_overflow = True
        
    if dec2 >= 1.0:
        dec2 = max_val
        input2_overflow = True
    elif dec2 < -1.0:
        dec2 = min_val
        input2_overflow = True
    
    # Convert decimals to SF binary format
    sf1 = decimal_to_sf(dec1, n_bits)
    sf2 = decimal_to_sf(dec2, n_bits)
    
    # Perform SF multiplication
    sf_result = sf_mul(sf1, sf2, n_bits)
    
    # Convert result back to decimal
    result_decimal = sf_to_decimal(sf_result)
    
    # Calculate exact decimal multiplication for comparison
    exact_decimal = original_dec1 * original_dec2
    
    return {
        'sf_binary': sf_result,
        'decimal': result_decimal,
        'inputs_sf': (sf1, sf2),
        'exact_decimal': exact_decimal,
        'error': abs(exact_decimal - result_decimal),
        'input1_overflow': input1_overflow,
        'input2_overflow': input2_overflow,
        'clamped_inputs': (dec1, dec2)
    }


def sf_add_dec(dec1: float, dec2: float, n_bits: int = 8) -> dict:
    """
    Adds two decimal numbers using SF arithmetic with overflow clamping.
    
    Args:
    - dec1, dec2: decimal numbers (will be clamped to SF range if needed)
    - n_bits: number of fractional bits for binary representation
    
    Returns:
    - Dictionary with result information
    """
    # Store originals and clamp inputs
    original_dec1, original_dec2 = dec1, dec2
    max_val = 1.0 - (2 ** (-n_bits))
    min_val = -1.0
    
    input1_overflow = False
    input2_overflow = False
    
    if dec1 >= 1.0:
        dec1 = max_val
        input1_overflow = True
    elif dec1 < -1.0:
        dec1 = min_val
        input1_overflow = True
        
    if dec2 >= 1.0:
        dec2 = max_val
        input2_overflow = True
    elif dec2 < -1.0:
        dec2 = min_val
        input2_overflow = True
    
    # Convert to SF format
    sf1 = decimal_to_sf(dec1, n_bits)
    sf2 = decimal_to_sf(dec2, n_bits)
    
    # Extract components
    sign1, frac1 = sf1[0], sf1[2:]
    sign2, frac2 = sf2[0], sf2[2:]
    
    # Convert to signed integers
    int1 = int(frac1, 2)
    int2 = int(frac2, 2)
    
    if sign1 == '1':
        int1 = -int1
    if sign2 == '1':
        int2 = -int2
    
    # Add
    result = int1 + int2
    
    # Handle overflow/underflow in SF integer space
    max_sf_int = (1 << n_bits) - 1  # Maximum positive SF integer
    min_sf_int = -(1 << n_bits)     # Minimum negative SF integer
    
    result_overflow = False
    if result > max_sf_int:
        result = max_sf_int
        result_overflow = True
    elif result < min_sf_int:
        result = min_sf_int
        result_overflow = True
    
    # Convert back to SF format
    abs_result = abs(result)
    sign_bit = '0' if result >= 0 else '1'
    frac_binary = bin(abs_result)[2:].zfill(n_bits)
    
    sf_result = f"{sign_bit}.{frac_binary}"
    result_decimal = sf_to_decimal(sf_result)
    exact_decimal = original_dec1 + original_dec2
    
    return {
        'sf_binary': sf_result,
        'decimal': result_decimal,
        'exact_decimal': exact_decimal,
        'error': abs(exact_decimal - result_decimal),
        'result_overflow': result_overflow,
        'input1_overflow': input1_overflow,
        'input2_overflow': input2_overflow
    }


def sf_tensor_mul(tensor_a: list, tensor_b: list, n_bits: int = 8) -> dict:
    """
    Multiplies two 2D tensors using signed fixed-point arithmetic with overflow clamping.
    
    Args:
    - tensor_a, tensor_b: 2D lists representing matrices (values will be clamped to SF range)
    - n_bits: number of fractional bits for SF representation
    
    Returns:
    - Dictionary with SF result, decimal result, exact result, and error analysis
    """
    # Validate dimensions
    rows_a = len(tensor_a)
    cols_a = len(tensor_a[0]) if rows_a > 0 else 0
    rows_b = len(tensor_b)
    cols_b = len(tensor_b[0]) if rows_b > 0 else 0
    
    if cols_a != rows_b:
        raise ValueError(f"Cannot multiply matrices: {rows_a}x{cols_a} × {rows_b}x{cols_b}")
    
    # Initialize result matrices
    sf_result = [[None for _ in range(cols_b)] for _ in range(rows_a)]
    decimal_result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
    exact_result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
    clamped_exact_result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    total_error = 0.0
    max_error = 0.0
    input_overflow_count = 0
    result_overflow_count = 0
    
    max_sf_val = 1.0 - (2 ** (-n_bits))
    min_sf_val = -1.0
    
    # Perform matrix multiplication using SF arithmetic
    for i in range(rows_a):
        for j in range(cols_b):
            # Compute dot product of row i of A and column j of B
            sf_sum = 0.0  # Accumulate in decimal for intermediate sums
            exact_sum = 0.0
            
            for k in range(cols_a):
                # Track input overflows
                if tensor_a[i][k] >= 1.0 or tensor_a[i][k] < -1.0:
                    input_overflow_count += 1
                if tensor_b[k][j] >= 1.0 or tensor_b[k][j] < -1.0:
                    input_overflow_count += 1
                
                # SF multiplication (this handles input clamping internally)
                mul_result = sf_mul_dec(tensor_a[i][k], tensor_b[k][j], n_bits)
                
                # Add to running sum (in decimal space)
                sf_sum += mul_result['decimal']
                exact_sum += tensor_a[i][k] * tensor_b[k][j]
            
            # Clamp sf_sum to valid SF range
            original_sf_sum = sf_sum
            if sf_sum >= 1.0:
                sf_sum = max_sf_val
                result_overflow_count += 1
            elif sf_sum < -1.0:
                sf_sum = min_sf_val
                result_overflow_count += 1
            
            # Convert final sum to SF format
            sf_binary = decimal_to_sf(sf_sum, n_bits)
            sf_result[i][j] = sf_binary
            decimal_result[i][j] = sf_to_decimal(sf_binary)
            exact_result[i][j] = exact_sum
            
            # Clamp exact result for error comparison
            if exact_sum >= 1.0:
                clamped_exact_result[i][j] = max_sf_val
            elif exact_sum < -1.0:
                clamped_exact_result[i][j] = min_sf_val
            else:
                clamped_exact_result[i][j] = exact_sum
            
            # Track errors against clamped exact result
            error = abs(clamped_exact_result[i][j] - decimal_result[i][j])
            total_error += error
            max_error = max(max_error, error)
    
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


def print_tensor(tensor, title, precision=6):
    """Helper function to print tensors nicely"""
    print(f"{title}:")
    for row in tensor:
        formatted_row = []
        for val in row:
            if isinstance(val, str):  # SF binary format
                formatted_row.append(f"{val:>12}")
            else:  # Decimal format
                formatted_row.append(f"{val:>{precision+6}.{precision}f}")
        print("  [" + ", ".join(formatted_row) + "]")
    print()


# Test tensor multiplication
print("\n" + "="*60)
print("TENSOR MULTIPLICATION EXAMPLE")
print("="*60)

A = [
    [0.32, -0.75, 0.11],
    [-0.58, 0.94, -0.23],
    [0.67, -0.12, -0.81]
]

B = [
    [-0.44, 0.09, 0.65],
    [0.77, -0.36, -0.52],
    [-0.19, 0.84, 0.27]
]

print("Input Tensors:")
print_tensor(A, "Tensor A (Decimal)", 2)

# Convert input tensors to SF format for display
A_sf = [[decimal_to_sf(A[i][j], 16) for j in range(len(A[i]))] for i in range(len(A))]
B_sf = [[decimal_to_sf(B[i][j], 16) for j in range(len(B[i]))] for i in range(len(B))]

print_tensor(A_sf, "Tensor A (SF Binary)")
print_tensor(B, "Tensor B (Decimal)", 2)
print_tensor(B_sf, "Tensor B (SF Binary)")

# Perform SF tensor multiplication
result = sf_tensor_mul(A, B, n_bits=16)

print("Results:")
print_tensor(result['sf_result'], "SF Binary Result")
print_tensor(result['decimal_result'], "SF Decimal Result", 6)
print_tensor(result['exact_result'], "Exact Decimal Result", 6)
print_tensor(result['clamped_exact_result'], "Clamped Exact Result", 6)

print(f"Error Analysis (SF vs Clamped Exact):")
print(f"  Total Error: {result['total_error']:.8f}")
print(f"  Average Error: {result['average_error']:.8f}")
print(f"  Maximum Error: {result['max_error']:.8f}")
print(f"  Input Overflow Count: {result['input_overflow_count']}")
print(f"  Result Overflow Count: {result['result_overflow_count']}")
print(f"  Result Shape: {result['result_shape']}")