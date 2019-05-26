/* Implementations of CUDA test functions.
 *
 * * * * * * * * * * * *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019 Stephen Sorley
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * * * * * * * * * * * *
 */
#include "test.h"

#include "timer.h"

#include <cmath>
#include <algorithm> //for std::max
#include <cstdio>
#include <vector>

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
