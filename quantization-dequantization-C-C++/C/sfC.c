#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "cJSON.h"
#include "cJSON_Utils.h"
#include <string.h>

int Mantissa_Bits;
float MaxValue;

char *strndup(const char *str, size_t n) {

    size_t len = strnlen(str, n); // Calculate the minimum of n and the string length
    char *copy = (char *)malloc(len + 1); // Allocate memory for the new string
    if (!copy) {
        return NULL; // Return NULL if memory allocation fails
    }
    memcpy(copy, str, len); // Copy the string content
    copy[len] = '\0'; // Null-terminate the new string
    return copy;
}
void setMantissaBits(int value){
    if(value>=4 && value<=16){
        Mantissa_Bits=value;
    }else{
        Mantissa_Bits=16;
    }
}

typedef union {
    float f;
    uint32_t i;
} FloatUnion;

float superfloat_max_value() {
    int mantissa_bits= Mantissa_Bits;
    if (mantissa_bits < 4 || mantissa_bits > 16) {
        printf("Error: Mantissa size must be between 4 and 16.\n");
        return 0.0f;
    }

    // Calculate max mantissa and scale
    int max_mantissa = (1 << mantissa_bits) - 1;  // All mantissa bits set to 1
    int scale = 1 << mantissa_bits;              // 2^mantissa_bits

    // Maximum representable value
    return (float)max_mantissa / scale;
}

void printFloatBits(float value) {//Prints string of individual bit values for float variable specified

float superfloat_max_value() {
    int mantissa_bits=Mantissa_Bits;
    if (mantissa_bits < 4 || mantissa_bits > 16) {
        printf("Error: Mantissa size must be between 4 and 16.\n");
        return 0.0f;
    }

    // Calculate max mantissa and scale
    int max_mantissa = (1 << mantissa_bits) - 1;  // All mantissa bits set to 1
    int scale = 1 << mantissa_bits;              // 2^mantissa_bits

    // Maximum representable value
    return (float)max_mantissa / scale;
}
    FloatUnion u;
    u.f = value;  // Store the float in the union

    // Print the 32-bit representation of the float (IEEE 754 format)
    printf("Float value: %f\n", value);
    printf("Binary representation: ");

    for (int i = 31; i >= 0; i--) {
        printf("%d", (u.i >> i) & 1);  // Print each bit from MSB to LSB
    }
    printf("\n");
}

float encode_superfloat(float value/*, int mantissa_bits*/) {
    int mantissa_bits=Mantissa_Bits;
    if (mantissa_bits < 4 || mantissa_bits > 16) {
        printf("Error: Mantissa size must be between 4 and 16.\n");
        return 0.0f;
    }

    if (value <= -1 ) {
        return -MaxValue;
    }else if(value >= 1){
        return MaxValue;
    }

    // Extract sign bit
    uint32_t sign = (value < 0) ? 1 : 0;

    // Convert value to positive for encoding
    if (sign) value = -value;

    // Scale value to fit in the mantissa range
    uint32_t mantissa = (uint32_t)(value * (1 << mantissa_bits));  // Scale to fit mantissa size

    // Assemble the IEEE 754 bits: sign (1 bit), zero exponent (8 bits), and mantissa
    uint32_t ieee_bits = (sign << 31) | (mantissa << (23 - mantissa_bits));  // Shift mantissa to correct position

    // Reinterpret bits as a float
    float result;
    *((uint32_t*)&result) = ieee_bits;

    return result;
}

float decode_superfloat(float sf/*, int mantissa_bits*/) {
    int mantissa_bits=Mantissa_Bits;
    if (mantissa_bits < 4 || mantissa_bits > 16) {
        printf("Error: Mantissa size must be between 4 and 16.\n");
        return;
    }

    // Access the bits of the float
    uint32_t ieee_bits = *((uint32_t*)&sf);

    // Extract components
    uint32_t sign = (ieee_bits >> 31) & 1;
    uint32_t mantissa = (ieee_bits >> (23 - mantissa_bits)) & ((1 << mantissa_bits) - 1);  // Extract mantissa

    // Compute the value
    float value = mantissa / (float)(1 << mantissa_bits);  // Scale back to original range
    if (sign) value = -value;

    return value;
    printf("Decoded Value: %f\n", value);
}

// Define the structure for nested matrices
typedef struct NestedMatrix {
    bool is_scalar; // true if scalar, false if nested matrix
    union {
        float scalar_value;               // Scalar value if is_scalar is true
        struct {
            int num_rows;                 // Number of rows in the matrix
            int num_cols;                 // Number of columns in the matrix
            struct NestedMatrix ***data;  // Pointer to nested matrices
        } matrix;                         // Matrix structure
    } value;
} NestedMatrix;

// Recursive function to traverse and quantize the nested matrix
void quantize_nested_matrix(NestedMatrix *matrix) {
    if (matrix->is_scalar) {
        // Apply quantization to scalar value
        matrix->value.scalar_value = encode_superfloat(matrix->value.scalar_value);
        //printf("FloatBItsEncoded");
        //printFloatBits(matrix->value.scalar_value);
    } else {
        // Traverse the nested matrix
        for (int i = 0; i < matrix->value.matrix.num_rows; i++) {
            for (int j = 0; j < matrix->value.matrix.num_cols; j++) {
                quantize_nested_matrix(matrix->value.matrix.data[i][j]);
            }
        }
    }
}

void dequantize_nested_matrix(NestedMatrix *matrix) {
    if (matrix->is_scalar) {
        // Apply quantization to scalar value
        matrix->value.scalar_value = decode_superfloat(matrix->value.scalar_value);
    } else {
        // Traverse the nested matrix
        for (int i = 0; i < matrix->value.matrix.num_rows; i++) {
            for (int j = 0; j < matrix->value.matrix.num_cols; j++) {
                dequantize_nested_matrix(matrix->value.matrix.data[i][j]);
            }
        }
    }
}

// Helper function to create a scalar NestedMatrix
NestedMatrix *create_scalar(float value) {
    NestedMatrix *matrix = malloc(sizeof(NestedMatrix));
    matrix->is_scalar = true;
    matrix->value.scalar_value = value;
    return matrix;
}

// Helper function to create a nested matrix
NestedMatrix *create_nested_matrix(int rows, int cols) {
    NestedMatrix *matrix = malloc(sizeof(NestedMatrix));
    matrix->is_scalar = false;
    matrix->value.matrix.num_rows = rows;
    matrix->value.matrix.num_cols = cols;
    matrix->value.matrix.data = malloc(rows * sizeof(NestedMatrix **));
    for (int i = 0; i < rows; i++) {
        matrix->value.matrix.data[i] = malloc(cols * sizeof(NestedMatrix *));
    }
    return matrix;
}

// Helper function to free a NestedMatrix
void free_nested_matrix(NestedMatrix *matrix) {
    if (matrix->is_scalar) {
        free(matrix);
    } else {
        for (int i = 0; i < matrix->value.matrix.num_rows; i++) {
            for (int j = 0; j < matrix->value.matrix.num_cols; j++) {
                free_nested_matrix(matrix->value.matrix.data[i][j]);
            }
            free(matrix->value.matrix.data[i]);
        }
        free(matrix->value.matrix.data);
        free(matrix);
    }
}

// Helper function to print the nested matrix
void print_nested_matrix(NestedMatrix *matrix, int depth) {
    for (int i = 0; i < depth; i++) printf("  "); // Indentation
    if (matrix->is_scalar) {
        printf("%.5f\n", matrix->value.scalar_value);
        //printFloatBits(matrix->value.scalar_value);
    } else {
        printf("[\n");
        for (int i = 0; i < matrix->value.matrix.num_rows; i++) {
            for (int j = 0; j < depth + 1; j++) printf("  "); // Indentation
            for (int j = 0; j < matrix->value.matrix.num_cols; j++) {
                print_nested_matrix(matrix->value.matrix.data[i][j], depth + 1);
            }
        }
        for (int i = 0; i < depth; i++) printf("  "); // Indentation
        printf("]\n");
    }
}



// Helper function to read the entire file into a buffer
char *read_file(const char *file_path, size_t *file_size) {
    FILE *file = fopen(file_path, "rb");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *buffer = malloc(*file_size);
    if (!buffer) {
        perror("Memory allocation failed");
        fclose(file);
        return NULL;
    }

    fread(buffer, 1, *file_size, file);
    fclose(file);
    return buffer;
}

// Function to parse and print the safetensors file (Doesn't work, look in main)
void parse_safetensors(const char *file_path) {
    size_t file_size;
    char *file_data = read_file(file_path, &file_size);
    if (!file_data) {
        return;
    }

    // Locate the start of the JSON header
    size_t json_start = 0;
    while (json_start < file_size && (file_data[json_start] < 32 || file_data[json_start] > 126)) {
        json_start++;
    }

    if (json_start >= file_size) {
        fprintf(stderr, "Failed to locate JSON start.\n");
        free(file_data);
        return;
    }

    // Locate the end of the JSON header
    size_t json_end = json_start;
    while (json_end < file_size && file_data[json_end] != '\0') {
        json_end++;
    }

    if (json_end == json_start || json_end >= file_size) {
        fprintf(stderr, "Invalid safetensors file: JSON header not found.\n");
        free(file_data);
        return;
    }

    // Extract the JSON header
    char *json_header = strndup(file_data + json_start, json_end - json_start);
    if (!json_header) {
        fprintf(stderr, "Failed to allocate memory for JSON header.\n");
        free(file_data);
        return;
    }

    printf("Extracted JSON Header:\n%s\n", json_header);

    // Parse the JSON header
    cJSON *json = cJSON_Parse(json_header);
    free(json_header);

    if (!json) {
        fprintf(stderr, "Failed to parse JSON header: %s\n", cJSON_GetErrorPtr());
        free(file_data);
        return;
    }

    printf("Parsed JSON:\n%s\n", cJSON_Print(json));

    // Extract tensor metadata
    cJSON *keys = cJSON_GetObjectItemCaseSensitive(json, "tensors");
    if (!keys || !cJSON_IsObject(keys)) {
        fprintf(stderr, "No tensor metadata found in JSON.\n");
        cJSON_Delete(json);
        free(file_data);
        return;
    }

    printf("\nBinary Tensor Data:\n");
    cJSON *tensor_key;
    cJSON_ArrayForEach(tensor_key, keys) {
        const char *key = tensor_key->string;
        cJSON *metadata = cJSON_GetObjectItem(keys, key);

        if (metadata) {
            printf("Tensor: %s\n", key);

            // Get shape and data offset
            cJSON *shape = cJSON_GetObjectItem(metadata, "shape");
            cJSON *dtype = cJSON_GetObjectItem(metadata, "dtype");
            cJSON *data_offsets = cJSON_GetObjectItem(metadata, "data_offsets");

            if (shape && cJSON_IsArray(shape) && dtype && data_offsets) {
                printf("  Shape: [");
                cJSON *dim;
                cJSON_ArrayForEach(dim, shape) {
                    printf("%d ", dim->valueint);
                }
                printf("]\n");
                printf("  DType: %s\n", dtype->valuestring);

                int start_offset = cJSON_GetArrayItem(data_offsets, 0)->valueint;
                int end_offset = cJSON_GetArrayItem(data_offsets, 1)->valueint;

                printf("  Data Offset: %d to %d\n", start_offset, end_offset);

                // Read and print binary data as floats (assuming dtype = float32)
                if (strcmp(dtype->valuestring, "F32") == 0) {
                    float *tensor_data = (float *)(file_data + start_offset);
                    int count = (end_offset - start_offset) / sizeof(float);

                    printf("  Data: ");
                    for (int i = 0; i < count; i++) {
                        printf("%f ", tensor_data[i]);
                    }
                    printf("\n");
                } else {
                    printf("  Unsupported dtype: %s\n", dtype->valuestring);
                }
            }
        }
    }

    // Clean up
    cJSON_Delete(json);
    free(file_data);
}

int main() {

    /*
    setMantissaBits(8);
    MaxValue=superfloat_max_value(Mantissa_Bits);
    //printFloatBits(MaxValue);

    int sfSize=16;
    float number=0.25;
    printf("Superfloat Bit Size : %d\n",sfSize);
    //printFloatBits(decode_superfloat(encode_superfloat(number)));
    //printf("\n%f",decode_superfloat(encode_superfloat(number)));

     // Create a nested matrix structure
    NestedMatrix *root = create_nested_matrix(2, 2);

    // Populate the matrix with scalars and sub-matrices
    root->value.matrix.data[0][0] = create_scalar(0.255);
    root->value.matrix.data[0][1] = create_scalar(0.755);

    NestedMatrix *sub_matrix = create_nested_matrix(2, 1);
    sub_matrix->value.matrix.data[0][0] = create_scalar(1.75);
    sub_matrix->value.matrix.data[1][0] = create_scalar(0.823);
    root->value.matrix.data[1][0] = sub_matrix;

    root->value.matrix.data[1][1] = create_scalar(0.919);

    for(int i=0;i<2;i++){
        for(int j=0;j<100;j++){
            if(i!=1 && j!=0){
                root->value.matrix.data[i][j] = create_scalar(rand()%4);
            }
        }
    }

    // Print the original matrix
    printf("Original Matrix:\n");
    print_nested_matrix(root, 0);

    // Quantize, Dequantize the nested matrix
    quantize_nested_matrix(root);
    dequantize_nested_matrix(root);

    // Print the quantized matrix
    printf("\nQuantized Matrix:\n");
    print_nested_matrix(root, 0);

    // Free the memory
    free_nested_matrix(root);



    if (argc < 2) {
        fprintf(stderr, "Usage: %s <safetensors file>\n", argv[0]);
        return 1;
    }
    */
    parse_safetensors("C:/Users/anand/Downloads/model2.safetensors");// Can't read .safetensors JSON file header showing nested structure of parameters
    return 0;
}

