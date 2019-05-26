/* Implementation cuBLAS test executable.
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
#include "timer.h"

#include <cuda_runtime.h> // Need to include this header manually, since this is a pure C++ file, not a CUDA file.
#include <cublas_v2.h>

#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Error-checking code.
namespace {
    const char *strip_dir(const char *path) {
        const char *last_slash = strrchr(path, '/');
        if(!last_slash) {
            last_slash = strrchr(path, '\\');
        }
        return (last_slash)? last_slash + 1 : path;
    }

    const char *cublas_err_str(cublasStatus_t stat) {
        switch(stat) {
            case CUBLAS_STATUS_SUCCESS: return "success";
            case CUBLAS_STATUS_NOT_INITIALIZED:  return "cuBLAS library not initialized, or initialization failed";
            case CUBLAS_STATUS_ALLOC_FAILED: return "memory allocation inside cuBLAS failed";
            case CUBLAS_STATUS_INVALID_VALUE: return "invalid parameter passed to cuBLAS function";
            case CUBLAS_STATUS_ARCH_MISMATCH: return "cuBLAS function needs newer CUDA arch version than currently targeted one";
            case CUBLAS_STATUS_MAPPING_ERROR: return "access to GPU memory space failed (failed texture bind, probably)";
            case CUBLAS_STATUS_EXECUTION_FAILED: return "cuBLAS GPU kernel failed to execute";
            case CUBLAS_STATUS_INTERNAL_ERROR: return "internal cuBLAS op failed (probably a bad cudaMemcpyAsync call)";
            case CUBLAS_STATUS_NOT_SUPPORTED: return "the requested functionality is not supported";
            case CUBLAS_STATUS_LICENSE_ERROR: return "requested functionality requires a license, which is not present or out of date";
        }
        return "<UNKNOWN>";
    }
}
#define CUDA_ERRCHK(X) do {\
    cudaError_t stat = X;\
    if(stat != cudaSuccess) {\
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", strip_dir(__FILE__), __LINE__, cudaGetErrorString(stat));\
        throw int(EXIT_FAILURE);\
    }\
} while(0);

#define CUBLAS_ERRCHK(X) do {\
    cublasStatus_t stat = X;\
    if(stat != CUBLAS_STATUS_SUCCESS) {\
        std::fprintf(stderr, "cuBLAS error %s:%d: %s\n", strip_dir(__FILE__), __LINE__, cublas_err_str(stat));\
        throw int(EXIT_FAILURE);\
    }\
} while(0);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Helper functions.
namespace {
    size_t to_colmajor(size_t row_idx, size_t col_idx, size_t num_rows) {
        return col_idx*num_rows + row_idx;
    }

    void modify(cublasHandle_t handle, float *mat, size_t ldmat, size_t n, size_t p, size_t q, float alpha, float beta) {
        mat += to_colmajor(p,q,ldmat);
        CUBLAS_ERRCHK(cublasSscal(handle, int(n)-int(q), &alpha, mat, int(ldmat)));
        CUBLAS_ERRCHK(cublasSscal(handle, int(ldmat)-int(p), &beta, mat, 1));
    }
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Main test code.
// This was adapted from the example code provided in Section 1.3 of the cuBLAS manual.
int main(int argc, char *argv[]) {
    (void)argc; (void)argv;

    int    ret            = EXIT_SUCCESS;
    size_t num_rows       = 6;
    size_t num_cols       = 5;
    float *d_a            = nullptr;
    cublasHandle_t handle = nullptr;

    std::vector<float> a(num_rows * num_cols);

    try {
        // Create a cuBLAS context for this test.
        CUBLAS_ERRCHK(cublasCreate(&handle));

        // Make a test matrix filled with dummy data.
        for(size_t col_idx = 0; col_idx < num_cols; ++col_idx) {
            for(size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
                a[to_colmajor(row_idx, col_idx, num_rows)] = float(row_idx*num_rows + col_idx+1);
            }
        }

        // Allocate memory for test matrix on device.
        CUDA_ERRCHK(cudaMalloc((void**)&d_a, a.size()*sizeof(*d_a)));

        // Copy test matrix to device.
        CUBLAS_ERRCHK(cublasSetMatrix(int(num_rows), int(num_cols), int(sizeof(*d_a)), a.data(),
            int(num_rows), d_a, int(num_rows)));

        // Perform a couple scalar ops on the matrix.
        modify(handle, d_a, num_rows, num_cols, 1, 2, 16.0f, 12.0f);

        // Copy the matrix back to the host.
        CUBLAS_ERRCHK(cublasGetMatrix(int(num_rows), int(num_cols), int(sizeof(*d_a)), d_a,
            int(num_rows), a.data(), int(num_rows)));
    }catch(int code) {
        ret = code;
    }

    // If operation was success, print results.
    if(ret == EXIT_SUCCESS) {
        for(size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            for(size_t col_idx = 0; col_idx < num_cols; ++col_idx) {
                std::printf("%7.0f ", a[to_colmajor(row_idx, col_idx, num_rows)]);
            }
            std::printf("\n");
        }
    }

    // Clean up resources allocated on the device.
    if(d_a)    cudaFree(d_a);
    if(handle) cublasDestroy(handle);

    return ret;
}
