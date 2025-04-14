#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cJSON.h"
#include <stdint.h>

// Reads the file content into memory
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

// Parses and prints a safetensors file
void parse_safetensors(const char *file_path) {
    size_t file_size;
    char *file_data = read_file(file_path, &file_size);
    if (!file_data) {
        return;
    }

    // Read JSON header length (first 4 bytes, little-endian)
    if (file_size < 4) {
        fprintf(stderr, "Invalid safetensors file: too small.\n");
        free(file_data);
        return;
    }

    uint32_t json_header_length = *(uint32_t *)file_data;
    if (json_header_length + 4 > file_size) {
        fprintf(stderr, "Invalid safetensors file: JSON header length exceeds file size.\n");
        free(file_data);
        return;
    }

    // Extract and parse JSON header
    char *json_header = strndup(file_data + 4, json_header_length);
    cJSON *json = cJSON_Parse(json_header);
    free(json_header);

    if (!json) {
        fprintf(stderr, "Failed to parse JSON header.\n");
        free(file_data);
        return;
    }

    printf("JSON Header:\n%s\n", cJSON_Print(json));

    // Read tensor metadata
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

            // Extract shape and data offset
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

                if (start_offset < 4 + json_header_length || end_offset > file_size) {
                    fprintf(stderr, "Invalid data offsets for tensor: %s\n", key);
                    continue;
                }

                printf("  Data Offset: %d to %d\n", start_offset, end_offset);

                // Read and print binary data as floats (assuming dtype = float32)
                if (strcmp(dtype->valuestring, "float32") == 0) {
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

    cJSON_Delete(json);
    free(file_data);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <safetensors file>\n", argv[0]);
        return 1;
    }

    parse_safetensors(argv[1]);
    return 0;
}

