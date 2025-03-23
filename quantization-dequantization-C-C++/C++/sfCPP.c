#include <iostream>
#include <cstdint>
#include <cmath>
#include <vector>
#include <memory>

int Mantissa_Bits;
float MaxValue;

void setMantissaBits(int value) {
    if (value >= 4 && value <= 16) {
        Mantissa_Bits = value;
    } else {
        Mantissa_Bits = 16;
    }
}

union FloatUnion {
    float f;
    uint32_t i;
};

float superfloat_max_value() {
    int mantissa_bits = Mantissa_Bits;
    if (mantissa_bits < 4 || mantissa_bits > 16) {
        std::cerr << "Error: Mantissa size must be between 4 and 16." << std::endl;
        return 0.0f;
    }

    int max_mantissa = (1 << mantissa_bits) - 1; // All mantissa bits set to 1
    int scale = 1 << mantissa_bits;             // 2^mantissa_bits

    return static_cast<float>(max_mantissa) / scale;
}

void printFloatBits(float value) {
    FloatUnion u;
    u.f = value;

    std::cout << "Float value: " << value << std::endl;
    std::cout << "Binary representation: ";
    for (int i = 31; i >= 0; i--) {
        std::cout << ((u.i >> i) & 1);
    }
    std::cout << std::endl;
}

float encode_superfloat(float value) {
    int mantissa_bits = Mantissa_Bits;
    if (mantissa_bits < 4 || mantissa_bits > 16) {
        std::cerr << "Error: Mantissa size must be between 4 and 16." << std::endl;
        return 0.0f;
    }

    if (value <= -1) {
        return -MaxValue;
    } else if (value >= 1) {
        return MaxValue;
    }

    uint32_t sign = (value < 0) ? 1 : 0;
    if (sign) value = -value;

    uint32_t mantissa = static_cast<uint32_t>(value * (1 << mantissa_bits));
    uint32_t ieee_bits = (sign << 31) | (mantissa << (23 - mantissa_bits));

    float result;
    *reinterpret_cast<uint32_t*>(&result) = ieee_bits;

    return result;
}

float decode_superfloat(float sf) {
    int mantissa_bits = Mantissa_Bits;
    if (mantissa_bits < 4 || mantissa_bits > 16) {
        std::cerr << "Error: Mantissa size must be between 4 and 16." << std::endl;
        return 0.0f;
    }

    uint32_t ieee_bits = *reinterpret_cast<uint32_t*>(&sf);
    uint32_t sign = (ieee_bits >> 31) & 1;
    uint32_t mantissa = (ieee_bits >> (23 - mantissa_bits)) & ((1 << mantissa_bits) - 1);

    float value = mantissa / static_cast<float>(1 << mantissa_bits);
    if (sign) value = -value;

    return value;
}

class NestedMatrix {
public:
    bool is_scalar;
    float scalar_value;
    std::vector<std::vector<std::shared_ptr<NestedMatrix>>> data;

    NestedMatrix(float scalar) : is_scalar(true), scalar_value(scalar) {}

    NestedMatrix(int rows, int cols)
        : is_scalar(false), data(rows, std::vector<std::shared_ptr<NestedMatrix>>(cols)) {}

    void quantize() {
        if (is_scalar) {
            scalar_value = encode_superfloat(scalar_value);
        } else {
            for (auto& row : data) {
                for (auto& cell : row) {
                    cell->quantize();
                }
            }
        }
    }

    void dequantize() {
        if (is_scalar) {
            scalar_value = decode_superfloat(scalar_value);
        } else {
            for (auto& row : data) {
                for (auto& cell : row) {
                    cell->dequantize();
                }
            }
        }
    }

    void print(int depth = 0) const {
        for (int i = 0; i < depth; ++i) std::cout << "  ";
        if (is_scalar) {
            std::cout << scalar_value << std::endl;
        } else {
            std::cout << "[\n";
            for (const auto& row : data) {
                for (int i = 0; i < depth + 1; ++i) std::cout << "  ";
                for (const auto& cell : row) {
                    cell->print(depth + 1);
                }
            }
            for (int i = 0; i < depth; ++i) std::cout << "  ";
            std::cout << "]\n";
        }
    }
};

int main() {
    setMantissaBits(8);
    MaxValue = superfloat_max_value();

    auto root = std::make_shared<NestedMatrix>(2, 2);
    root->data[0][0] = std::make_shared<NestedMatrix>(0.255f);
    root->data[0][1] = std::make_shared<NestedMatrix>(0.755f);

    auto sub_matrix = std::make_shared<NestedMatrix>(2, 1);
    sub_matrix->data[0][0] = std::make_shared<NestedMatrix>(1.75f);
    sub_matrix->data[1][0] = std::make_shared<NestedMatrix>(0.823f);
    root->data[1][0] = sub_matrix;

    root->data[1][1] = std::make_shared<NestedMatrix>(0.919f);

    std::cout << "Original Matrix:\n";
    root->print();

    root->quantize();
    root->dequantize();

    std::cout << "\nQuantized Matrix:\n";
    root->print();

    return 0;
}
