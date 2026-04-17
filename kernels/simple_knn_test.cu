// simple_knn_test.cu — deterministic smoke test for standalone CUDA KNN

#include "splat_data.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static float exact_mean_3nn_dist2(const float* xyz, int N, int i) {
    std::vector<float> d2;
    d2.reserve(N - 1);
    for (int j = 0; j < N; j++) {
        if (i == j) continue;
        float dx = xyz[i*3]   - xyz[j*3];
        float dy = xyz[i*3+1] - xyz[j*3+1];
        float dz = xyz[i*3+2] - xyz[j*3+2];
        d2.push_back(dx*dx + dy*dy + dz*dz);
    }
    std::sort(d2.begin(), d2.end());
    int k = std::min(3, (int)d2.size());
    float sum = 0.f;
    for (int n = 0; n < k; n++) sum += d2[n];
    return k > 0 ? sum / (float)k : 0.f;
}

int main() {
    printf("=== simple_knn test ===\n\n");

    const int N = 5;
    float xyz[N * 3] = {
        0.f,  0.f, 0.f,
        1.f,  0.f, 0.f,
        0.f,  1.f, 0.f,
        0.f,  0.f, 1.f,
        10.f, 0.f, 0.f,
    };

    std::vector<float> got = simple_knn::dist_cuda2_host(xyz, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = exact_mean_3nn_dist2(xyz, N, i);
        float err = fabsf(got[i] - expected);
        bool row_ok = err < 1e-4f;
        ok = ok && row_ok;
        printf("point %d  mean_dist2=%.6f  expected=%.6f  %s\n",
               i, got[i], expected, row_ok ? "OK" : "FAIL");
    }

    if (!ok) {
        fprintf(stderr, "\nKNN test failed.\n");
        return 1;
    }

    printf("\nAll tests passed.\n");
    return 0;
}

