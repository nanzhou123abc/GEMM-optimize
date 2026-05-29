#include <cstdint>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <string>
#include <arm_neon.h>   // float16_t
#include <arm_bf16.h>   // __bf16 / bfloat16_t (ARM BF16 扩展)

namespace w4a8 {
    constexpr int AWQ_ORDER[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    constexpr int AWQ_REVERSE_ORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    constexpr int BIT_SHIFTS[8] = {0, 4, 8, 12, 16, 20, 24, 28};

    struct LinearParams {
        int M;          // batch size（token 数）
        int K;          // 输入特征维度
        int N;          // 输出特征维度
        int group_size; // 量化 group 大小
        int groups;     // group 数量 = K / group_size
    };
    inline LinearParams check_shapes(int x_rows, int x_cols,           // x:       [M, K]
                                    int qw_rows, int qw_cols,         // qweight: [K, N/8]
                                    int qz_rows, int qz_cols,         // qzeros:  [groups, N/8]
                                    int sc_rows, int sc_cols) {          // scales:  [groups, N]
        int K = x_cols;
        int M = x_rows;
        
        int n_over_8 = qw_cols;
        int N = n_over_8 * 8;
        
        if (qw_rows != K) {
            throw std::runtime_error("qweight rows must equal K");
        }
        if (qz_cols != n_over_8) {
            throw std::runtime_error(
                "qzeros N dim mismatch: " + std::to_string(qz_cols) +
                " vs " + std::to_string(n_over_8));
        }
        if (qz_rows != sc_rows) {
            throw std::runtime_error(
                "qzeros and scales group dim mismatch: " +
                std::to_string(qz_rows) + " vs " + std::to_string(sc_rows));
        }
        if (sc_cols != N) {
            throw std::runtime_error(
                "scales N dim mismatch: " + std::to_string(sc_cols) +
                " vs " + std::to_string(N));
        }
        if (K % qz_rows != 0) {
            throw std::runtime_error(
                "K=" + std::to_string(K) +
                " not divisible by groups=" + std::to_string(qz_rows));
        }

        int group_size = K / qz_rows;
        int groups = qz_rows;
        return {M, K, N, group_size, groups};
    }
    std::pair<std::vector<int8_t>, std::vector<float>> quantize_activation_per_token(const __bf16* x, int M, int K) {
        std::vector<int8_t> x_q(M * K);
        std::vector<float> x_scale(M);

        std::vector<float> max_abs(M, 0.0f);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                float v = static_cast<float>(x[i * K + j]);   // bf16 -> fp32
                max_abs[i] = std::max(max_abs[i], std::abs(v));
            }
            if(max_abs[i] == 0.0f) {
                max_abs[i] = 1.0f;
            }
            x_scale[i] = max_abs[i] / 127.0f;
            float inv_scale = 1.0f / x_scale[i];
            for (int j = 0; j < K; j++) {
                float v = static_cast<float>(x[i * K + j]) * inv_scale;
                v = std::round(v);
                v = std::max(-128.0f, std::min(127.0f, v));
                x_q[i * K + j] = static_cast<int8_t>(v);
            }
        }
        return {x_q, x_scale};
    }

    std::vector<int8_t> unpack_int4_along_n_awq(const int32_t* packed, int rows, int n_over_8) {
        int N = n_over_8 * 8;
        std::vector<int8_t> result(rows * N);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < n_over_8; j++) {
                int32_t val = packed[i * n_over_8 + j]; 
                int8_t nibbles[8];
                //取出8个4bit
                for(int k = 0; k < 8; k++) {
                    nibbles[k] = (val >> (k * 4)) & 0xF;   //(k * 4) = BIT_SHIFTS[k]
                }
                //按照AWQ_REVERSE_ORDER的顺序写入result
                for(int k = 0; k < 8; k++) {
                    result[i * N + j * 8 + k] = nibbles[AWQ_REVERSE_ORDER[k]];
                }
            }
        }
        return result;
    }
    std::vector<int8_t> unpack_awq_qweight(const int32_t* packed, int K, int n_over_8) {
        return unpack_int4_along_n_awq(packed, K, n_over_8);
    }
    std::vector<int8_t> unpack_awq_qzeros(const int32_t* packed, int K, int n_over_8) {
        return unpack_int4_along_n_awq(packed, K, n_over_8);
    }

    void gemm_int32(const int32_t* a, const int32_t* b, int32_t* c, int m, int k, int n) {
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                int32_t sum = 0;
                for(int kk = 0; kk < k; kk++) {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }


    std::vector<float> w4a8_linear(const __bf16* x, int M, int K,
                                const int32_t* qweight, int N, 
                                const int32_t* qzeros, int group_size, 
                                const float16_t* scales, const __bf16* bias) {

        int n_over_8 = N / 8;
        int groups = K / group_size;
        auto params = check_shapes(M, K, K, n_over_8, groups, n_over_8, groups, N);
        if(M == 0 || K == 0) {
            return std::vector<float>(M * N, 0.0f);
        }
        auto [x_q, x_scale] = quantize_activation_per_token(x, M, K);
        std::vector<int8_t> w_q = unpack_awq_qweight(qweight, K, n_over_8);
        std::vector<int8_t> w_z = unpack_awq_qzeros(qzeros, groups, n_over_8);

        std::vector<float> out_fp32(M * N, 0.0f);
       
        for(int g = 0; g < groups; g++) {
            int k0 = g * group_size;
            // int k1 = (g + 1) * group_size;
            std::vector<int32_t> w_block(group_size * N);
            for(int k = 0; k < group_size; k++) {
                for(int n = 0; n < N; n++) {
                    w_block[k * N + n] = static_cast<int32_t>(w_q[(k0 + k) * N + n]) - static_cast<int32_t>(w_z[g * N + n]);
                }
            }

            std::vector<int32_t> x_block(M * group_size);
            for(int m = 0; m < M; m++) {
                for(int k = 0; k < group_size; k++) {
                    x_block[m * group_size + k] = static_cast<int32_t>(x_q[m * K + k0 + k]);
                }
            }
                //x_block = x_q[:, k0:k1].to(torch.int32) 
            std::vector<int32_t> acc(M * N);
            gemm_int32(x_block.data(), w_block.data(), acc.data(), M, group_size, N);

            for(int m = 0; m < M; m++) {
                for(int n = 0; n < N; n++) {
                    out_fp32[m * N + n] += static_cast<float>(acc[m * N + n]) * static_cast<float>(scales[g * N + n]);
                }
            }
        }
        for(int m = 0; m < M; m++) {
            for(int n = 0; n < N; n++) {
                out_fp32[m * N + n] *= x_scale[m];
            }
        }
        if(bias != nullptr) {
            if(bias.size() != N) {
                throw std::runtime_error("bias 形状应为: " + std::to_string(bias.size()) + " 当前： " + std::to_string(N));
            }
            for(int m = 0; m < M; m++) {
                for(int n = 0; n < N; n++) {
                    out_fp32[m * N + n] += static_cast<float>(bias[n]);
                }
            }
        }
        return out_fp32;
    }
}