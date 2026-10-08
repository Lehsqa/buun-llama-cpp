#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "quants.h"

#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static bool test_direct_parity() {
    std::mt19937 rng(0x49324d54);
    const auto * q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);

    for (const auto type : {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S}) {
        const auto * traits = ggml_get_type_traits_cpu(type);
        const auto dot_mt = type == GGML_TYPE_IQ2_S ? ggml_vec_dot_iq2_s_q8_K_mt : ggml_vec_dot_iq3_s_q8_K_mt;
        for (const int k : {256, 512, 2560}) {
            std::vector<uint8_t> weights(ggml_row_size(type, k));
            std::vector<float> input(k);
            std::vector<block_q8_K> activations[8];
            const void * pointers[8];
            for (int token = 0; token < 8; ++token) {
                activations[token].resize(k / QK_K);
                pointers[token] = activations[token].data();
            }
            for (int trial = 0; trial < 64; ++trial) {
                for (auto & byte : weights) {
                    byte = uint8_t(rng());
                }
                for (int block = 0; block < k / QK_K; ++block) {
                    const ggml_fp16_t d = ggml_fp32_to_fp16(float(int(rng() % 4097) - 2048) / 1024.0f);
                    std::memcpy(weights.data() + block * ggml_type_size(type), &d, sizeof(d));
                }
                for (int pattern = 0; pattern < 3; ++pattern) {
                    for (int token = 0; token < 8; ++token) {
                        for (int j = 0; j < k; ++j) {
                            input[j] = pattern == 0 ? float(int(rng() % 16385) - 8192) / 1024.0f :
                                       pattern == 1 ? 0.0f : ((j + token) % 2 ? -8.0f : 8.0f);
                        }
                        q8->from_float(input.data(), activations[token].data(), k);
                    }
                    for (int ncols = 1; ncols <= 8; ++ncols) {
                        float actual[8];
                        dot_mt(k, actual, weights.data(), pointers, ncols);
                        for (int token = 0; token < ncols; ++token) {
                            float expected;
                            traits->vec_dot(k, &expected, 0, weights.data(), 0, pointers[token], 0, 1);
                            if (std::memcmp(&expected, &actual[token], sizeof(float)) != 0) {
                                std::fprintf(stderr, "%s k=%d trial=%d pattern=%d ncols=%d token=%d: expected=%a actual=%a\n",
                                             ggml_type_name(type), k, trial, pattern, ncols, token, expected, actual[token]);
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }
    std::puts("IQ2_S/IQ3_S multi-token direct parity passed");
    return true;
}

static bool test_graph_parity(ggml_backend_t backend, enum ggml_type type, enum ggml_type input_type,
                              int threads, int nt, int slots, bool window) {
    constexpr int k = 2560, m = 19, experts = 16, used = 10, lo = 7;
    ggml_backend_cpu_set_n_threads(backend, threads);
    ggml_init_params params = { 2 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, type, k, m, experts);
    ggml_tensor * storage = ggml_new_tensor_3d(ctx, input_type, k, slots + 1, nt);
    ggml_tensor * input = ggml_view_3d(ctx, storage, k, slots, nt, storage->nb[1], storage->nb[2], storage->nb[1]);
    ggml_tensor * ids_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used + 3, nt);
    ggml_tensor * ids = ggml_view_2d(ctx, ids_storage, used, nt, ids_storage->nb[1], sizeof(int32_t));
    ggml_tensor * out = ggml_mul_mat_id(ctx, weights, input, ids);
    if (window) {
        ggml_mul_mat_id_set_expert_window(out, lo, experts);
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "graph buffer allocation failed\n");
        ggml_free(ctx);
        return false;
    }

    std::mt19937 rng(0x47524150);
    std::vector<uint8_t> weight_data(ggml_nbytes(weights));
    for (auto & byte : weight_data) {
        byte = uint8_t(rng());
    }
    for (size_t offset = 0; offset < weight_data.size(); offset += ggml_type_size(type)) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(float(int(rng() % 4097) - 2048) / 1024.0f);
        std::memcpy(weight_data.data() + offset, &d, sizeof(d));
    }
    ggml_backend_tensor_set(weights, weight_data.data(), 0, weight_data.size());

    const auto * q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const auto * traits = ggml_get_type_traits_cpu(type);
    std::vector<float> float_data(k * (slots + 1) * nt);
    for (auto & value : float_data) {
        value = float(int(rng() % 16385) - 8192) / 1024.0f;
    }
    std::vector<uint8_t> quantized(ggml_row_size(GGML_TYPE_Q8_K, k) * (slots + 1) * nt);
    for (int row = 0; row < (slots + 1) * nt; ++row) {
        q8->from_float(float_data.data() + row * k,
                       quantized.data() + row * ggml_row_size(GGML_TYPE_Q8_K, k), k);
    }
    if (input_type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(storage, float_data.data(), 0, ggml_nbytes(storage));
    } else {
        ggml_backend_tensor_set(storage, quantized.data(), 0, ggml_nbytes(storage));
    }
    std::vector<int32_t> routes((used + 3) * nt, -1);
    for (int token = 0; token < nt; ++token) {
        for (int slot = 0; slot < used; ++slot) {
            routes[token * (used + 3) + slot + 1] = window && slot == used - 1 ? lo - 1 :
                (window ? lo : 0) + (nt == 1 ? slot : (slot + token) % 4);
        }
    }
    ggml_backend_tensor_set(ids_storage, routes.data(), 0, ggml_nbytes(ids_storage));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    bool ok = status == GGML_STATUS_SUCCESS;
    std::vector<float> actual(m * used * nt);
    if (ok) {
        ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
        for (int token = 0; token < nt && ok; ++token) {
            for (int slot = 0; slot < used && ok; ++slot) {
                const int expert = routes[token * (used + 3) + slot + 1] - (window ? lo : 0);
                const void * activation = quantized.data() +
                    (token * (slots + 1) + 1 + slot % slots) * ggml_row_size(GGML_TYPE_Q8_K, k);
                for (int row = 0; row < m; ++row) {
                    float expected = 0.0f;
                    if (expert >= 0) {
                        traits->vec_dot(k, &expected, 0,
                            weight_data.data() + (expert * m + row) * ggml_row_size(type, k),
                            0, activation, 0, 1);
                    }
                    const float value = actual[(token * used + slot) * m + row];
                    if (std::memcmp(&expected, &value, sizeof(float)) != 0) {
                        std::fprintf(stderr, "%s input=%s threads=%d nt=%d slots=%d window=%d token=%d slot=%d row=%d: expected=%a actual=%a\n",
                            ggml_type_name(type), ggml_type_name(input_type), threads, nt, slots, window,
                            token, slot, row, expected, value);
                        ok = false;
                        break;
                    }
                }
            }
        }
    } else {
        std::fprintf(stderr, "graph compute failed: %d\n", int(status));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

static void benchmark() {
    constexpr int k = 2560, rows_per_expert = 640;
    constexpr size_t corpus_size = size_t(256) * 1024 * 1024;
    std::mt19937 rng(0x42454e43);
    const auto * q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    std::vector<float> input(k);
    std::vector<block_q8_K> activations[8];
    const void * pointers[8];
    for (int token = 0; token < 8; ++token) {
        activations[token].resize(k / QK_K);
        for (auto & value : input) {
            value = float(int(rng() % 16385) - 8192) / 1024.0f;
        }
        q8->from_float(input.data(), activations[token].data(), k);
        pointers[token] = activations[token].data();
    }
    volatile double checksum = 0.0;
    for (const auto type : {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S}) {
        const auto * traits = ggml_get_type_traits_cpu(type);
        const auto dot_mt = type == GGML_TYPE_IQ2_S ? ggml_vec_dot_iq2_s_q8_K_mt : ggml_vec_dot_iq3_s_q8_K_mt;
        const size_t row_size = ggml_row_size(type, k);
        const size_t experts = corpus_size / (rows_per_expert * row_size);
        const size_t rows = experts * rows_per_expert;
        std::vector<uint8_t> weights(rows * row_size);
        for (auto & byte : weights) {
            byte = uint8_t(rng());
        }
        for (size_t offset = 0; offset < weights.size(); offset += ggml_type_size(type)) {
            const ggml_fp16_t d = ggml_fp32_to_fp16(float(int(rng() % 4097) - 2048) / 1024.0f);
            std::memcpy(weights.data() + offset, &d, sizeof(d));
        }
        std::printf("%s k=%d rows/expert=%d corpus=%zu bytes experts=%zu\n",
                    ggml_type_name(type), k, rows_per_expert, weights.size(), experts);
        for (const int width : {1, 2, 3, 4, 8}) {
            const auto run = [&](bool multi) {
                double sum = 0.0;
                const auto start = std::chrono::steady_clock::now();
                for (size_t row = 0; row < rows; ++row) {
                    float dots[8];
                    const void * weight = weights.data() + row * row_size;
                    if (multi) {
                        dot_mt(k, dots, weight, pointers, width);
                    } else {
                        for (int token = 0; token < width; ++token) {
                            traits->vec_dot(k, &dots[token], 0, weight, 0, pointers[token], 0, 1);
                        }
                    }
                    for (int token = 0; token < width; ++token) {
                        sum += dots[token];
                    }
                }
                const auto end = std::chrono::steady_clock::now();
                checksum = checksum + sum;
                return std::chrono::duration<double, std::milli>(end - start).count() / experts;
            };
            run(false);
            run(true);
            std::vector<double> scalar(5), multi(5);
            for (int pass = 0; pass < 5; ++pass) {
                if (pass % 2 == 0) {
                    scalar[pass] = run(false);
                    multi[pass] = run(true);
                } else {
                    multi[pass] = run(true);
                    scalar[pass] = run(false);
                }
            }
            std::sort(scalar.begin(), scalar.end());
            std::sort(multi.begin(), multi.end());
            std::printf("  tokens=%d vec_dot=%.3f ms mt=%.3f ms per 640-row projection speedup=%.3fx\n",
                        width, scalar[2], multi[2], scalar[2] / multi[2]);
        }
    }
    std::printf("checksum=%a\n", double(checksum));
}

int main(int argc, char ** argv) {
    ggml_cpu_init();
    if (!test_direct_parity()) {
        return 1;
    }
    if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
        benchmark();
        return 0;
    }
    if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--bench]\n", argv[0]);
        return 1;
    }
    // The scalar oracle requires the existing panel optimization to be disabled.
#ifdef _WIN32
    _putenv_s("GGML_NO_IQ_PANEL", "1");
#else
    setenv("GGML_NO_IQ_PANEL", "1", 1);
#endif
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        return 1;
    }
    bool ok = true;
    for (const auto type : {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S}) {
        for (const auto input_type : {GGML_TYPE_F32, GGML_TYPE_Q8_K}) {
            for (const int threads : {1, 6, 12}) {
                for (const int nt : {1, 2, 3, 4, 8, 9}) {
                    for (const int slots : {1, 10}) {
                        for (const bool window : {false, true}) {
                            ok = test_graph_parity(backend, type, input_type, threads, nt, slots, window) && ok;
                        }
                    }
                }
            }
        }
    }
    ggml_backend_free(backend);
    if (ok) {
        std::puts("IQ2_S/IQ3_S CPU graph bitwise parity passed");
    }
    return ok ? 0 : 1;
}
