// GGML_SCHED_TIMING=N prints one "sched-timing:" summary every N graph computes.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<std::string> g_lines;

static void capture(ggml_log_level level, const char * text, void * user_data) {
    (void) level; (void) user_data;
    if (strstr(text, "sched-timing:") != nullptr) {
        g_lines.emplace_back(text);
    }
}

int main() {
#ifdef _WIN32
    _putenv_s("GGML_SCHED_TIMING", "1");
#else
    setenv("GGML_SCHED_TIMING", "1", 1);
#endif
    ggml_backend_load_all();
    ggml_log_set(capture, nullptr);

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) { fprintf(stderr, "no CPU backend\n"); return 1; }

    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_set_input(a); ggml_set_input(b);
    ggml_tensor * c = ggml_add(ctx, a, b);
    ggml_set_output(c);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) { fprintf(stderr, "alloc failed\n"); return 1; }

    std::vector<float> ones(64, 1.0f);
    for (int it = 0; it < 2; ++it) {
        ggml_backend_tensor_set(a, ones.data(), 0, ggml_nbytes(a));
        ggml_backend_tensor_set(b, ones.data(), 0, ggml_nbytes(b));
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
    }

    ggml_log_set(nullptr, nullptr);
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(cpu);

    if (g_lines.size() != 2) { fprintf(stderr, "expected 2 sched-timing lines, got %zu\n", g_lines.size()); return 1; }
    for (const auto & l : g_lines) {
        if (l.find("graphs=1 ") == std::string::npos || l.find("cmp_ms[") == std::string::npos || l.find("CPU=") == std::string::npos) {
            fprintf(stderr, "bad line: %s", l.c_str()); return 1;
        }
    }
    printf("OK\n");
    return 0;
}
