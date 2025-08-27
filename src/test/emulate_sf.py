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




# Example usage:
# print(sf_mul('0.101', '0.110'))  # Should return '0.011'
# print(sf_mul('1.101', '0.110'))  # Should return '1.011'
# print(sf_mul('1.101', '1.110'))  # Should return '0.011'

print(sf_mul('0.1110001','0.0001001'))