#include "llama-ple-reader.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

#define REQUIRE(c) do { if (!(c)) { fprintf(stderr, "%s:%d: REQUIRE(%s) failed\n", __FILE__, __LINE__, #c); return 1; } } while (0)

static const uint64_t HEAD = 1000;  // unaligned tensor offset
static const size_t   ROW  = 120;   // q5_1 row of 160 values, as the shipped table
static const uint64_t NROW = 5000;

static std::vector<uint8_t> make_file(const fs::path & p) {
    std::mt19937 gen(42);
    std::vector<uint8_t> all(HEAD + NROW*ROW + 37);
    for (auto & b : all) b = uint8_t(gen());
    std::ofstream(p, std::ios::binary).write((const char *) all.data(), all.size());
    return all;
}

static int check_rows(llama_ple_reader & r, const std::vector<uint8_t> & all, const std::vector<int32_t> & rows) {
    std::vector<uint8_t> out(rows.size()*ROW);
    r.read_rows(rows.data(), rows.size(), out.data());
    for (size_t i = 0; i < rows.size(); ++i) {
        REQUIRE(memcmp(out.data() + i*ROW, all.data() + HEAD + (uint64_t) rows[i]*ROW, ROW) == 0);
    }
    return 0;
}

static int run_case(const fs::path & p, const std::vector<uint8_t> & all, bool direct, uint32_t threads, uint32_t cache) {
    llama_ple_reader_params rp;
    rp.path = p.string(); rp.offset = HEAD; rp.n_rows = NROW; rp.row_size = ROW;
    rp.n_threads = threads; rp.cache_rows = cache; rp.direct = direct;
    llama_ple_reader r(rp);

    std::mt19937 gen(7);
    std::vector<int32_t> rows(20000);
    for (auto & x : rows) x = int32_t(gen() % NROW);
    // rows straddling a 4 KiB page boundary, and the last row
    for (uint64_t k = 0; k < NROW; ++k) {
        const uint64_t o = HEAD + k*ROW;
        if (o/4096 != (o + ROW - 1)/4096) rows.push_back(int32_t(k));
    }
    rows.push_back(int32_t(NROW - 1));
    if (check_rows(r, all, rows)) return 1;
    // second pass: with a cache big enough, every row is a hit
    const auto s0 = r.stats();
    if (check_rows(r, all, rows)) return 1;
    const auto s1 = r.stats();
    if (cache >= NROW*8) REQUIRE(s1.hits - s0.hits == rows.size());
    if (cache == 0)      REQUIRE(s1.hits == 0);
    // prefetch then read
    std::vector<int32_t> pf = { 1, 2, 3, 4000 };
    r.prefetch(pf.data(), pf.size());
    r.wait_idle();
    if (check_rows(r, all, pf)) return 1;
    // out of range throws
    int32_t bad = int32_t(NROW);
    uint8_t tmp[ROW];
    bool threw = false;
    try { r.read_rows(&bad, 1, tmp); } catch (const std::runtime_error &) { threw = true; }
    REQUIRE(threw);
    printf("case direct=%d(%d) threads=%u cache=%u: %s\n", direct, r.direct(), threads, cache, r.stats_str().c_str());
    return 0;
}

// rows -> f32 must equal ggml_get_rows on the CPU backend, bit for bit
static int run_dequant(const fs::path & p) {
    const int64_t ne0 = 160, nr = 64;
    std::mt19937 gen(3);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    std::vector<float> src(ne0*nr);
    for (auto & v : src) v = dis(gen);
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q5_1, ne0)*nr);
    ggml_quantize_chunk(GGML_TYPE_Q5_1, src.data(), q.data(), 0, nr, ne0, nullptr);
    std::ofstream(p, std::ios::binary).write((const char *) q.data(), q.size());

    llama_ple_reader_params rp;
    rp.path = p.string(); rp.offset = 0; rp.n_rows = nr; rp.row_size = ggml_row_size(GGML_TYPE_Q5_1, ne0);
    llama_ple_reader r(rp);
    std::vector<int32_t> rows = { 5, 0, 63, 5, 17 };
    std::vector<float> got(rows.size()*ne0);
    r.read_rows_f32(rows.data(), rows.size(), GGML_TYPE_Q5_1, ne0, got.data());

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    REQUIRE(cpu != nullptr);
    ggml_init_params ip = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_1, ne0, nr);
    ggml_tensor * ix = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rows.size());
    ggml_tensor * g  = ggml_get_rows(ctx, t, ix);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, g);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
    ggml_backend_tensor_set(ix, rows.data(), 0, rows.size()*sizeof(int32_t));
    REQUIRE(ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS);
    std::vector<float> want(rows.size()*ne0);
    ggml_backend_tensor_get(g, want.data(), 0, want.size()*sizeof(float));
    REQUIRE(memcmp(got.data(), want.data(), want.size()*sizeof(float)) == 0);
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(cpu);
    return 0;
}

int main() {
    ggml_backend_load_all();
    // cwd is the build tree (usually a real filesystem, so O_DIRECT works); the temp dir may be tmpfs (fallback path)
    const fs::path here = fs::current_path() / "test-ple-reader.bin";
    const fs::path tmp  = fs::temp_directory_path() / "test-ple-reader.bin";
    for (const auto & p : { here, tmp }) {
        const auto all = make_file(p);
        if (run_case(p, all, true,  16, 1u << 16)) return 1;
        if (run_case(p, all, true,  1,  0))        return 1;
        if (run_case(p, all, false, 4,  64))       return 1;
        fs::remove(p);
    }
    bool threw = false;
    try { llama_ple_reader_params rp; rp.path = "/nonexistent/x"; rp.n_rows = 1; rp.row_size = 8; llama_ple_reader r(rp); }
    catch (const std::runtime_error &) { threw = true; }
    REQUIRE(threw);
    const fs::path dq = fs::current_path() / "test-ple-reader-q51.bin";
    if (run_dequant(dq)) return 1;
    fs::remove(dq);
    printf("OK\n");
    return 0;
}
