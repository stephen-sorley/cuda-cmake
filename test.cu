#include "test.h"

#include <cmath>
#include <algorithm> //for std::max
#include <cstdio>
#include <chrono>
#include <vector>

namespace {
    using clock  = std::chrono::steady_clock ;
    using timept = std::chrono::time_point<clock>;
    static double elapsed_sec(const timept& start) {
        return std::chrono::duration<double>(clock::now() - start).count();
    }
}

__global__
void add(size_t N, float *x, float *y) {
    size_t thread_grid_idx     = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    for(size_t i = thread_grid_idx; i<N; i += num_threads_in_grid) {
        y[i] = x[i] + y[i];
    }
}

void test_add(size_t N) {
    float *d_x = nullptr;
    float *d_y = nullptr;

    cudaMalloc(&d_x, N*sizeof(*d_x));
    cudaMalloc(&d_y, N*sizeof(*d_y));

    std::vector<float> x(N, 1.0f);
    std::vector<float> y(N, 2.0f);

    // Copy vectors to device
    timept start = clock::now();
    cudaMemcpy(d_x, x.data(), N*sizeof(*d_x), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), N*sizeof(*d_y), cudaMemcpyHostToDevice);

    // Perform the addition.
    const int num_sm            = 20; // GTX 1080
    const int blocks_per_sm     = 16;
    const int threads_per_block = 1024;
    add<<<blocks_per_sm*num_sm, threads_per_block>>>(N, d_x, d_y);

    // Copy results back from device.
    cudaMemcpy(y.data(), d_y, N*sizeof(*d_y), cudaMemcpyDeviceToHost);
    printf("Elapsed: %0.3f (s) for %g adds\n", elapsed_sec(start), double(N));

    cudaFree(d_x);
    cudaFree(d_y);

    // Verify that returned results are OK.
    float max_err = 0.0f;
    for(size_t i=0; i<N; ++i) {
        max_err = std::max(max_err, std::abs(y[i] - 3.0f));
    }
    std::printf("[ADD] Max error: %g\n", max_err);
}
