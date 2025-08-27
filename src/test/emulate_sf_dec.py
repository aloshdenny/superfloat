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
    Converts a decimal number in range [-1, 1) to signed fixed-point binary format.
    
    Args:
    - decimal_val: decimal number between -1 and 1
    - n_bits: number of fractional bits
    
    Returns:
    - Binary string in s.xxx format
    """
    if not (-1 <= decimal_val < 1):
        raise ValueError("Decimal value must be in range [-1, 1)")
    
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
    Multiplies two decimal numbers in SF range [-1, 1) and returns result in both formats.
    
    Args:
    - dec1, dec2: decimal numbers between -1 and 1
    - n_bits: number of fractional bits for binary representation
    
    Returns:
    - Dictionary with 'sf_binary', 'decimal', 'inputs_sf', and 'exact_decimal' keys
    """
    # Validate inputs
    if not (-1 <= dec1 < 1) or not (-1 <= dec2 < 1):
        raise ValueError("Both decimal values must be in range [-1, 1)")
    
    # Convert decimals to SF binary format
    sf1 = decimal_to_sf(dec1, n_bits)
    sf2 = decimal_to_sf(dec2, n_bits)
    
    # Perform SF multiplication
    sf_result = sf_mul(sf1, sf2, n_bits)
    
    # Convert result back to decimal
    result_decimal = sf_to_decimal(sf_result)
    
    # Calculate exact decimal multiplication for comparison
    exact_decimal = dec1 * dec2
    
    return {
        'sf_binary': sf_result,
        'decimal': result_decimal,
        'inputs_sf': (sf1, sf2),
        'exact_decimal': exact_decimal,
        'error': abs(exact_decimal - result_decimal)
    }

result = sf_mul_dec(0.5, 0.213, 16)
print(f"  SF inputs: {result['inputs_sf'][0]} × {result['inputs_sf'][1]}")
print(f"  SF result: {result['sf_binary']}")
print(f"  Decimal result: {result['decimal']:.6f}")
print(f"  Exact decimal: {result['exact_decimal']:.6f}")
print(f"  Error: {result['error']:.6f}")