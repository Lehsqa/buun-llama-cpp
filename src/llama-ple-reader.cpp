#include "llama-ple-reader.h"
#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#define LLAMA_PLE_PREAD 1
#endif

static constexpr uint64_t PLE_ALIGN    = 4096;      // O_DIRECT offset/length/buffer alignment
static constexpr uint64_t PLE_JOB_MAX  = 64*1024;   // merge neighbouring pages into one read up to this size
static constexpr uint32_t PLE_WAYS     = 8;
static constexpr int      PLE_LAT_RING = 4096;
static constexpr int      PLE_LOG_S    = 30;        // stats line at most every 30 s

static int64_t ple_now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

namespace {

// one aligned read covering the pages of several requested rows
struct ple_job {
    uint64_t start = 0;          // aligned file offset
    uint64_t len   = 0;          // aligned length
    std::vector<std::pair<int32_t, uint8_t *>> rows; // row id, destination (nullptr = cache only)
};

// a batch of jobs a caller waits for
struct ple_batch {
    std::mutex              mtx;
    std::condition_variable cv;
    size_t                  pending = 0;
    std::exception_ptr      error;
};

#ifdef LLAMA_PLE_PREAD
// 4 KiB-aligned scratch buffer for O_DIRECT reads
struct ple_buf {
    uint8_t * p = nullptr;
    explicit ple_buf(size_t n) {
        void * q = nullptr;
        if (posix_memalign(&q, PLE_ALIGN, n) != 0) throw std::bad_alloc();
        p = (uint8_t *) q;
    }
    ~ple_buf() { free(p); }
    ple_buf(const ple_buf &) = delete;
    ple_buf & operator=(const ple_buf &) = delete;
};
#endif

} // namespace

struct llama_ple_reader::impl {
    llama_ple_reader_params prm;
    int  fd = -1;
    bool is_direct = false;

    // row cache: n_sets x PLE_WAYS, striped locks
    uint64_t n_sets = 0;
    std::vector<int64_t>  tags;   // -1 = empty
    std::vector<uint8_t>  data;
    std::vector<uint8_t>  victim; // round-robin per set
    std::vector<std::mutex> locks;

    // workers
    std::mutex mtx;
    std::condition_variable cv, cv_idle;
    std::deque<std::pair<ple_job, std::shared_ptr<ple_batch>>> queue;
    size_t busy = 0;
    bool stop = false;
    std::vector<std::thread> workers;

    // stats
    std::atomic<uint64_t> n_rows{0}, n_hits{0}, n_reads{0}, n_bytes{0}, wait_us{0};
    std::mutex lat_mtx;
    std::vector<uint32_t> lat = std::vector<uint32_t>(PLE_LAT_RING, 0);
    uint64_t lat_n = 0;
    std::atomic<int64_t> last_log_us{0};

    std::mutex & lock_of(uint64_t set) { return locks[set % locks.size()]; }
    uint64_t set_of(int32_t row) const { return ((uint64_t) row * 0x9E3779B97F4A7C15ull) % n_sets; }

    bool cache_get(int32_t row, uint8_t * out) {
        if (n_sets == 0) return false;
        const uint64_t s = set_of(row);
        std::lock_guard<std::mutex> lk(lock_of(s));
        for (uint32_t w = 0; w < PLE_WAYS; ++w) {
            if (tags[s*PLE_WAYS + w] == row) {
                memcpy(out, data.data() + (s*PLE_WAYS + w)*prm.row_size, prm.row_size);
                return true;
            }
        }
        return false;
    }

    void cache_put(int32_t row, const uint8_t * src) {
        if (n_sets == 0) return;
        const uint64_t s = set_of(row);
        std::lock_guard<std::mutex> lk(lock_of(s));
        for (uint32_t w = 0; w < PLE_WAYS; ++w) {
            if (tags[s*PLE_WAYS + w] == row) return;
        }
        const uint32_t w = victim[s]++ % PLE_WAYS;
        tags[s*PLE_WAYS + w] = row;
        memcpy(data.data() + (s*PLE_WAYS + w)*prm.row_size, src, prm.row_size);
    }

    void record_latency(uint64_t us) {
        std::lock_guard<std::mutex> lk(lat_mtx);
        lat[lat_n++ % PLE_LAT_RING] = (uint32_t) std::min<uint64_t>(us, UINT32_MAX);
    }

    void run_job(const ple_job & job) {
#ifdef LLAMA_PLE_PREAD
        ple_buf buf(job.len);
        const int64_t t0 = ple_now_us();
        uint64_t got = 0;
        while (got < job.len) {
            const ssize_t r = pread(fd, buf.p + got, job.len - got, (off_t) (job.start + got));
            if (r < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error(std::string("PLE pread failed: ") + strerror(errno));
            }
            if (r == 0) break; // end of file: the rows must still be covered, checked below
            got += (uint64_t) r;
        }
        record_latency((uint64_t) (ple_now_us() - t0));
        n_reads++;
        n_bytes += got;
        for (const auto & [row, dst] : job.rows) {
            const uint64_t o = prm.offset + (uint64_t) row*prm.row_size - job.start;
            if (o + prm.row_size > got) {
                throw std::runtime_error("PLE read ended before the row (file truncated?)");
            }
            cache_put(row, buf.p + o);
            if (dst) memcpy(dst, buf.p + o, prm.row_size);
        }
#else
        GGML_UNUSED(job);
        throw std::runtime_error("PLE direct reads are not supported on this platform");
#endif
    }

    void worker() {
        for (;;) {
            std::pair<ple_job, std::shared_ptr<ple_batch>> item;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&] { return stop || !queue.empty(); });
                if (stop && queue.empty()) return;
                item = std::move(queue.front());
                queue.pop_front();
                busy++;
            }
            std::exception_ptr err;
            try { run_job(item.first); } catch (...) { err = std::current_exception(); }
            if (item.second) {
                std::lock_guard<std::mutex> lk(item.second->mtx);
                if (err && !item.second->error) item.second->error = err;
                if (--item.second->pending == 0) item.second->cv.notify_all();
            }
            {
                std::lock_guard<std::mutex> lk(mtx);
                busy--;
                if (queue.empty() && busy == 0) cv_idle.notify_all();
            }
        }
    }

    // group the requested (row, dst) pairs into aligned jobs, rows sorted by offset
    std::vector<ple_job> make_jobs(std::vector<std::pair<int32_t, uint8_t *>> & req) const {
        std::sort(req.begin(), req.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        std::vector<ple_job> jobs;
        for (const auto & r : req) {
            const uint64_t o  = prm.offset + (uint64_t) r.first*prm.row_size;
            const uint64_t s  = o / PLE_ALIGN * PLE_ALIGN;
            const uint64_t e  = (o + prm.row_size + PLE_ALIGN - 1) / PLE_ALIGN * PLE_ALIGN;
            if (!jobs.empty() && s <= jobs.back().start + jobs.back().len && e - jobs.back().start <= PLE_JOB_MAX) {
                jobs.back().len = std::max(jobs.back().len, e - jobs.back().start);
            } else {
                jobs.push_back({ s, e - s, {} });
            }
            jobs.back().rows.push_back(r);
        }
        return jobs;
    }

    void maybe_log() {
        const int64_t now = ple_now_us();
        int64_t last = last_log_us.load();
        if (now - last >= (int64_t) PLE_LOG_S*1000000 && last_log_us.compare_exchange_strong(last, now)) {
            LLAMA_LOG_INFO("ple-reader: %s\n", owner_stats_str().c_str());
        }
    }

    std::function<std::string()> owner_stats_str;
};

llama_ple_reader::llama_ple_reader(const llama_ple_reader_params & params) : pimpl(std::make_unique<impl>()) {
    auto & d = *pimpl;
    d.prm = params;
    if (params.row_size == 0 || params.row_size > PLE_ALIGN || params.n_rows == 0) {
        throw std::runtime_error("PLE reader: row size must be in (0, 4096] and the table non-empty");
    }
#ifdef LLAMA_PLE_PREAD
#ifdef O_DIRECT
    if (params.direct) {
        d.fd = open(params.path.c_str(), O_RDONLY | O_DIRECT);
        d.is_direct = d.fd >= 0;
        if (d.is_direct) {
            // some filesystems (FUSE, network, >4 KiB logical blocks) accept O_DIRECT at open and fail the read:
            // probe one aligned page; any error (a short read at EOF is fine) means use buffered reads instead
            ple_buf probe(PLE_ALIGN);
            ssize_t r;
            do {
                r = pread(d.fd, probe.p, PLE_ALIGN, (off_t) (params.offset / PLE_ALIGN * PLE_ALIGN));
            } while (r < 0 && errno == EINTR);
            if (r < 0 || params.test_fail_direct_probe) {
                close(d.fd);
                d.fd = -1;
                d.is_direct = false;
            }
        }
    }
#endif
    if (d.fd < 0) {
        d.fd = open(params.path.c_str(), O_RDONLY);
#ifdef POSIX_FADV_RANDOM
        if (d.fd >= 0) posix_fadvise(d.fd, 0, 0, POSIX_FADV_RANDOM);
#endif
    }
    if (d.fd < 0) {
        throw std::runtime_error("PLE reader: cannot open " + params.path + ": " + strerror(errno));
    }
#else
    throw std::runtime_error("PLE reader: positioned reads are not supported on this platform");
#endif
    if (params.cache_rows > 0) {
        d.n_sets = std::max<uint64_t>(1, params.cache_rows / PLE_WAYS);
        d.tags.assign(d.n_sets*PLE_WAYS, -1);
        d.data.resize(d.n_sets*PLE_WAYS*params.row_size);
        d.victim.assign(d.n_sets, 0);
        d.locks = std::vector<std::mutex>(64);
    }
    d.owner_stats_str = [this] { return stats_str(); };
    const uint32_t nt = std::max<uint32_t>(1, params.n_threads);
    for (uint32_t i = 0; i < nt; ++i) {
        d.workers.emplace_back([&d] { d.worker(); });
    }
}

llama_ple_reader::~llama_ple_reader() {
    auto & d = *pimpl;
    {
        std::lock_guard<std::mutex> lk(d.mtx);
        d.stop = true;
    }
    d.cv.notify_all();
    for (auto & t : d.workers) t.join();
#ifdef LLAMA_PLE_PREAD
    if (d.fd >= 0) close(d.fd);
#endif
}

void llama_ple_reader::read_rows(const int32_t * rows, size_t n, uint8_t * out) {
    auto & d = *pimpl;
    const int64_t t0 = ple_now_us();
    std::vector<std::pair<int32_t, uint8_t *>> miss;
    for (size_t i = 0; i < n; ++i) {
        if (rows[i] < 0 || (uint64_t) rows[i] >= d.prm.n_rows) {
            throw std::runtime_error("PLE reader: row " + std::to_string(rows[i]) + " out of range");
        }
        if (d.cache_get(rows[i], out + i*d.prm.row_size)) {
            d.n_hits++;
        } else {
            miss.emplace_back(rows[i], out + i*d.prm.row_size);
        }
    }
    d.n_rows += n;
    if (!miss.empty()) {
        auto jobs = d.make_jobs(miss);
        auto b = std::make_shared<ple_batch>();
        b->pending = jobs.size();
        {
            std::lock_guard<std::mutex> lk(d.mtx);
            for (auto & j : jobs) d.queue.emplace_back(std::move(j), b);
        }
        d.cv.notify_all();
        std::unique_lock<std::mutex> lk(b->mtx);
        b->cv.wait(lk, [&] { return b->pending == 0; });
        if (b->error) std::rethrow_exception(b->error);
    }
    d.wait_us += (uint64_t) (ple_now_us() - t0);
    d.maybe_log();
}

void llama_ple_reader::read_rows_f32(const int32_t * rows, size_t n, ggml_type type, int64_t ne0, float * out) {
    GGML_ASSERT(ggml_row_size(type, ne0) == pimpl->prm.row_size);
    std::vector<uint8_t> raw(n*pimpl->prm.row_size);
    read_rows(rows, n, raw.data());
    if (type == GGML_TYPE_F32) {
        memcpy(out, raw.data(), raw.size());
        return;
    }
    const auto to_float = ggml_get_type_traits(type)->to_float;
    GGML_ASSERT(to_float != nullptr);
    for (size_t i = 0; i < n; ++i) {
        to_float(raw.data() + i*pimpl->prm.row_size, out + i*ne0, ne0);
    }
}

void llama_ple_reader::prefetch(const int32_t * rows, size_t n) {
    auto & d = *pimpl;
    if (d.n_sets == 0) return;
    std::vector<std::pair<int32_t, uint8_t *>> req;
    std::vector<uint8_t> scratch(d.prm.row_size);
    for (size_t i = 0; i < n; ++i) {
        if (rows[i] < 0 || (uint64_t) rows[i] >= d.prm.n_rows) continue;
        if (!d.cache_get(rows[i], scratch.data())) req.emplace_back(rows[i], nullptr);
    }
    if (req.empty()) return;
    auto jobs = d.make_jobs(req);
    {
        std::lock_guard<std::mutex> lk(d.mtx);
        for (auto & j : jobs) d.queue.emplace_back(std::move(j), nullptr);
    }
    d.cv.notify_all();
}

void llama_ple_reader::wait_idle() {
    auto & d = *pimpl;
    std::unique_lock<std::mutex> lk(d.mtx);
    d.cv_idle.wait(lk, [&] { return d.queue.empty() && d.busy == 0; });
}

bool   llama_ple_reader::direct()   const { return pimpl->is_direct; }
size_t llama_ple_reader::row_size() const { return pimpl->prm.row_size; }

llama_ple_reader_stats llama_ple_reader::stats() const {
    auto & d = *pimpl;
    llama_ple_reader_stats s;
    s.rows = d.n_rows; s.hits = d.n_hits; s.reads = d.n_reads; s.bytes = d.n_bytes; s.wait_us = d.wait_us;
    std::vector<uint32_t> v;
    {
        std::lock_guard<std::mutex> lk(d.lat_mtx);
        v.assign(d.lat.begin(), d.lat.begin() + std::min<uint64_t>(d.lat_n, PLE_LAT_RING));
    }
    if (!v.empty()) {
        std::sort(v.begin(), v.end());
        s.lat_p50_us = v[v.size()/2];
        s.lat_p99_us = v[std::min(v.size() - 1, v.size()*99/100)];
    }
    return s;
}

std::string llama_ple_reader::stats_str() const {
    const auto s = stats();
    char buf[256];
    snprintf(buf, sizeof(buf), "%s rows=%llu hit=%.1f%% reads=%llu MiB=%.1f p50=%lluus p99=%lluus blocked_ms=%.1f",
            pimpl->is_direct ? "direct" : "buffered", (unsigned long long) s.rows, s.rows ? 100.0*s.hits/s.rows : 0.0,
            (unsigned long long) s.reads, s.bytes/1048576.0, (unsigned long long) s.lat_p50_us,
            (unsigned long long) s.lat_p99_us, s.wait_us/1000.0);
    return buf;
}
