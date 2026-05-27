#pragma once
// Per-kernel-bucket timing + NVTX annotations for M3 perf analysis.
// See notes/perf_plan.md for the high-level plan.
//
// Usage in train.cu:
//
//   #include "kernels/profiling.cuh"
//   ...
//   prof::registry().enabled = (iter >= cfg.profile_start_iter)
//                              && (iter <  cfg.profile_start_iter + cfg.profile_iters);
//   {
//       PROFILE_SCOPE("rasterize_fwd");
//       launch_rasterize_fwd(...);
//   }
//   ...
//   prof::registry().flush_iter();
//
// Cost when disabled: one boolean check + one NVTX push/pop (~ns).
// Cost when enabled : one cudaEventRecord per scope + one
// cudaDeviceSynchronize() per iter inside flush_iter().

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#if __has_include(<nvtx3/nvToolsExt.h>)
  #include <nvtx3/nvToolsExt.h>
  #define PROFILE_HAS_NVTX 1
#elif __has_include(<nvToolsExt.h>)
  #include <nvToolsExt.h>
  #define PROFILE_HAS_NVTX 1
#else
  #define PROFILE_HAS_NVTX 0
#endif

namespace prof {

struct BucketStats {
    double   sum_ms    = 0.0;
    double   sum_sq_ms = 0.0;
    uint64_t calls     = 0;

    inline void add(double ms) {
        sum_ms    += ms;
        sum_sq_ms += ms * ms;
        calls     += 1;
    }
    inline double mean() const { return calls ? sum_ms / (double)calls : 0.0; }
    inline double stddev() const {
        if (calls < 2) return 0.0;
        double m   = mean();
        double var = sum_sq_ms / (double)calls - m * m;
        return var > 0.0 ? std::sqrt(var) : 0.0;
    }
};

struct PendingEvent {
    const char* name;
    cudaEvent_t start;
    cudaEvent_t stop;
};

class Registry {
public:
    bool enabled        = false;
    int  measured_iters = 0;
    int  N_start        = 0;
    int  N_end          = 0;

    void submit(const char* name, cudaEvent_t start, cudaEvent_t stop) {
        if (!enabled) return;
        pending_.push_back({name, start, stop});
    }

    // Compute elapsed-ms for every submitted pair this iter, accumulate into
    // per-bucket stats, recycle the events. One device sync per iter.
    void flush_iter() {
        if (!enabled || pending_.empty()) return;
        cudaDeviceSynchronize();
        for (auto& p : pending_) {
            float ms = 0.f;
            cudaEventElapsedTime(&ms, p.start, p.stop);
            stats_[p.name].add((double)ms);
            free_.push_back(p.start);
            free_.push_back(p.stop);
        }
        pending_.clear();
        measured_iters++;
    }

    cudaEvent_t acquire() {
        if (!free_.empty()) {
            cudaEvent_t e = free_.back();
            free_.pop_back();
            return e;
        }
        cudaEvent_t e;
        cudaEventCreate(&e);
        return e;
    }

    void print_summary(FILE* fp = stdout) const {
        if (measured_iters == 0 || stats_.empty()) {
            std::fprintf(fp, "[profile] no measured iterations\n");
            return;
        }
        double total_mean = 0.0;
        for (auto& kv : stats_) total_mean += kv.second.mean();

        std::fprintf(fp,
            "\n=== kernel breakdown  (mean over %d iters; N_start=%d N_end=%d) ===\n",
            measured_iters, N_start, N_end);
        std::fprintf(fp, "%-22s %10s %10s %8s %8s\n",
                     "bucket", "mean_ms", "std_ms", "%iter", "calls");
        for (auto& kv : stats_) {
            const auto& s = kv.second;
            double pct = total_mean > 0.0 ? 100.0 * s.mean() / total_mean : 0.0;
            std::fprintf(fp, "%-22s %10.4f %10.4f %7.2f%% %8llu\n",
                         kv.first.c_str(), s.mean(), s.stddev(), pct,
                         (unsigned long long)s.calls);
        }
        std::fprintf(fp, "%-22s %10.4f\n", "TOTAL", total_mean);
    }

    void append_csv(const std::string& path,
                    const std::string& tag,
                    int W, int H) const {
        if (path.empty() || measured_iters == 0) return;

        bool need_header = false;
        {
            std::ifstream probe(path);
            need_header = !probe.good();
        }
        std::ofstream out(path, std::ios::app);
        if (!out) {
            std::fprintf(stderr, "[profile] failed to open %s for write\n",
                         path.c_str());
            return;
        }
        if (need_header) {
            out << "tag,bucket,mean_ms,std_ms,calls,iters,N_start,N_end,W,H\n";
        }
        for (auto& kv : stats_) {
            const auto& s = kv.second;
            out << tag << ',' << kv.first << ','
                << s.mean()  << ',' << s.stddev() << ',' << s.calls << ','
                << measured_iters << ',' << N_start << ',' << N_end << ','
                << W << ',' << H << '\n';
        }
    }

    void reset() {
        for (auto& p : pending_) {
            free_.push_back(p.start);
            free_.push_back(p.stop);
        }
        pending_.clear();
        stats_.clear();
        measured_iters = 0;
        N_start = N_end = 0;
    }

private:
    std::vector<PendingEvent> pending_;
    std::vector<cudaEvent_t>  free_;
    std::map<std::string, BucketStats> stats_;
};

inline Registry& registry() {
    static Registry r;
    return r;
}

// RAII wrapper. Constructor pushes an NVTX range + (if enabled) records a
// start event; destructor records the stop event and submits the pair.
class Scope {
public:
    explicit Scope(const char* name) : name_(name) {
#if PROFILE_HAS_NVTX
        nvtxRangePushA(name);
#endif
        if (registry().enabled) {
            active_ = true;
            start_  = registry().acquire();
            stop_   = registry().acquire();
            cudaEventRecord(start_);
        }
    }
    ~Scope() {
        if (active_) {
            cudaEventRecord(stop_);
            registry().submit(name_, start_, stop_);
        }
#if PROFILE_HAS_NVTX
        nvtxRangePop();
#endif
    }
    Scope(const Scope&)            = delete;
    Scope& operator=(const Scope&) = delete;

private:
    const char* name_;
    bool        active_ = false;
    cudaEvent_t start_  = nullptr;
    cudaEvent_t stop_   = nullptr;
};

} // namespace prof

#define PROF_CONCAT_INNER(a, b) a##b
#define PROF_CONCAT(a, b)       PROF_CONCAT_INNER(a, b)
#define PROFILE_SCOPE(name)     ::prof::Scope PROF_CONCAT(_prof_scope_, __LINE__)(name)
