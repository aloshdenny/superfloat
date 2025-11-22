import ast
import re
import sys


def parse_matrix_file(filename):
    """
    Parse a matrix from a text file.
    Supports formats:
    - Python list notation with hex values: [[0x1234, 0x5678], ...]
    - Python list notation with decimal values: [[1.5, 2.3], ...]
    - Space/comma separated values
    
    Returns:
        tuple: (matrix as list of lists, n, is_hex)
    """
    with open(filename, 'r') as f:
        content = f.read().strip()
    
    try:
        # Try to parse as Python literal (handles [[0x..., 0x...], ...] format)
        matrix = ast.literal_eval(content)
        if not isinstance(matrix, list) or not matrix:
            raise ValueError("Invalid matrix format")
        
        n = len(matrix)
        if not all(isinstance(row, list) and len(row) == n for row in matrix):
            raise ValueError("Matrix must be square (n×n)")
        
        # Check if values are hex integers or floats/decimals
        first_val = matrix[0][0]
        is_hex = isinstance(first_val, int)
        
        # If decimal values, convert to hex (assuming they're in range -1 to 1)
        if not is_hex:
            matrix = convert_decimal_to_hex(matrix)
            is_hex = True
            
        return matrix, n, is_hex
        
    except (ValueError, SyntaxError):
        # Try to parse as space/comma separated values
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        matrix = []
        
        for line in lines:
            # Split by comma or whitespace
            values = re.split(r'[,\s]+', line)
            row = []
            for val in values:
                if val.startswith('0x') or val.startswith('0X'):
                    row.append(int(val, 16))
                else:
                    try:
                        row.append(int(val, 16))  # Try hex without 0x prefix
                    except ValueError:
                        row.append(int(float(val)))  # Try as decimal
            matrix.append(row)
        
        n = len(matrix)
        if not all(len(row) == n for row in matrix):
            raise ValueError("Matrix must be square (n×n)")
        
        return matrix, n, True


def convert_decimal_to_hex(matrix):
    """
    Convert decimal matrix (values typically -1 to 1) to hex integers.
    Uses a simple mapping for demonstration - adjust as needed.
    """
    n = len(matrix)
    hex_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            # Map float to 16-bit signed integer
            # This is a simplified conversion - adjust based on your needs
            val = int((matrix[i][j] + 1) * 32767.5)  # Map [-1,1] to [0, 65535]
            val = max(0, min(65535, val))  # Clamp to 16-bit range
            row.append(val)
        hex_matrix.append(row)
    return hex_matrix


def extract_diagonals(matrix, hex_digits=4):
    """
    Extract diagonals from n×n matrix and pad with zeros.
    Diagonals go from top-right to bottom-left.
    
    Args:
        matrix: n×n matrix of integer values
        hex_digits: number of hex digits per element (default 4 for 16-bit values)
    
    Returns:
        list of diagonals, each padded to length n
    """
    n = len(matrix)
    diagonals = []
    padding_str = "0" * hex_digits
    
    # Total number of diagonals in an n×n matrix is (2n-1)
    for d in range(2 * n - 1):
        diagonal = []
        
        # For each diagonal, find all elements that belong to it
        for i in range(n):
            j = d - i
            if 0 <= j < n:
                # Convert to hex string with proper padding
                val = format(matrix[i][j], f'0{hex_digits}X')
                diagonal.append(val)
        
        # Pad with zeros to make length n
        # Diagonals 0 to n-1: pad on the left
        # Diagonals n to 2n-2: pad on the right
        padding_needed = n - len(diagonal)
        if d < n:
            padded_diagonal = [padding_str] * padding_needed + diagonal
        else:
            padded_diagonal = diagonal + [padding_str] * padding_needed
        
        diagonals.append(padded_diagonal)
    
    return diagonals


def extract_diagonals_reversed(matrix, hex_digits=4):
    """
    Extract diagonals from n×n matrix with elements in reverse order.
    Same diagonals as extract_diagonals, but each diagonal is reversed.
    
    Args:
        matrix: n×n matrix of integer values
        hex_digits: number of hex digits per element (default 4 for 16-bit values)
    
    Returns:
        list of reversed diagonals, each padded to length n
    """
    n = len(matrix)
    diagonals = []
    padding_str = "0" * hex_digits
    
    # Total number of diagonals in an n×n matrix is (2n-1)
    for d in range(2 * n - 1):
        diagonal = []
        
        # Extract diagonal elements (same as first function)
        for i in range(n):
            j = d - i
            if 0 <= j < n:
                val = format(matrix[i][j], f'0{hex_digits}X')
                diagonal.append(val)
        
        # Reverse the diagonal elements
        diagonal = diagonal[::-1]
        
        # Pad with zeros to make length n
        padding_needed = n - len(diagonal)
        if d < n:
            padded_diagonal = [padding_str] * padding_needed + diagonal
        else:
            padded_diagonal = diagonal + [padding_str] * padding_needed
        
        diagonals.append(padded_diagonal)
    
    return diagonals


def process_matrices(file1, file2, output_file=None):
    """
    Process two matrix files and generate diagonal stream output.
    
    Args:
        file1: path to first matrix file (B matrix)
        file2: path to second matrix file (A matrix)
        output_file: optional output file path (if None, prints to stdout)
    """
    # Parse both matrices
    matrix1, n1, _ = parse_matrix_file(file1)
    matrix2, n2, _ = parse_matrix_file(file2)
    
    if n1 != n2:
        raise ValueError(f"Matrix sizes don't match: {n1}×{n1} vs {n2}×{n2}")
    
    n = n1
    hex_digits = 4  # 16-bit values
    
    # Extract diagonals from both matrices
    diagonals1 = extract_diagonals(matrix1, hex_digits)
    diagonals2 = extract_diagonals_reversed(matrix2, hex_digits)
    
    # Calculate total bits
    total_bits = n * hex_digits * 4  # n elements × hex_digits × 4 bits per hex digit
    
    # Generate output
    output_lines = []
    
    for i in range(2 * n - 1):
        output_lines.append(f"A={total_bits}'h{''.join(diagonals2[i])};")
        output_lines.append(f"B={total_bits}'h{''.join(diagonals1[i])};")
        output_lines.append("#2")
    
    # Write output
    output_text = '\n'.join(output_lines)
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_text)
        print(f"Output written to {output_file}")
    else:
        print(output_text)


if __name__ == "__main__":
    # Hardcoded file paths
    file1 = "superfloat/src/test/B_8x8_hex.txt"  # B matrix
    file2 = "superfloat/src/test/A_8x8_hex.txt"  # A matrix
    output_file = "superfloat/src/test/8x8_hex_streamed.txt"  # Set to a filename to save output, or None to print to console

    try:
        process_matrices(file1, file2, output_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)