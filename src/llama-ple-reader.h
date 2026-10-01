#pragma once

// Positioned-read row reader for huge, sparsely read tables (the PLE n-gram table: ~16 scattered ~120 B rows per token
// out of ~37 GiB). Bypasses the page cache with O_DIRECT where the filesystem allows it; keeps a small row cache.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "ggml.h"

struct llama_ple_reader_params {
    std::string path;
    uint64_t    offset     = 0;        // tensor data offset in the file
    uint64_t    n_rows     = 0;
    size_t      row_size   = 0;        // bytes per row, <= 4096
    uint32_t    n_threads  = 16;
    uint32_t    cache_rows = 1u << 20; // 0 = no row cache
    bool        direct     = true;     // try O_DIRECT, fall back to buffered pread
    bool        test_fail_direct_probe = false; // tests: treat the O_DIRECT probe as refused
};

struct llama_ple_reader_stats {
    uint64_t rows = 0, hits = 0, reads = 0, bytes = 0, wait_us = 0, lat_p50_us = 0, lat_p99_us = 0;
};

class llama_ple_reader {
public:
    explicit llama_ple_reader(const llama_ple_reader_params & params); // throws std::runtime_error
    ~llama_ple_reader();
    void read_rows(const int32_t * rows, size_t n, uint8_t * out);                                   // blocking, out: n*row_size
    void read_rows_f32(const int32_t * rows, size_t n, ggml_type type, int64_t ne0, float * out);  // as CPU ggml_get_rows
    void prefetch(const int32_t * rows, size_t n);                                                  // async cache warm
    void wait_idle();                                                                                // tests: drain prefetch
    bool   direct() const;
    size_t row_size() const;
    llama_ple_reader_stats stats() const;
    std::string stats_str() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
