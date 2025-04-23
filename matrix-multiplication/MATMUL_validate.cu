#include "MATMUL.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void transpose(const float* in, float* out, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out[j * rows + i] = in[i * cols + j];
        }
    }
}

extern "C" int do_validate(const struct parameters *p)
{
    float *M = p->M;
    float *N = p->N;
    float *P = p->P;

    int r_M = p->r_M, c_M = p->c_M;
    int r_N = p->r_N, c_N = p->c_N;
    int r_P = p->r_P, c_P = p->c_P;

    // Check dimension validity
    if (c_M != r_N || r_M != r_P || c_N != c_P)
    {
        fprintf(stderr, "Matrix dimensions don't match for multiplication.\n");
        return 0;
    }

    // Allocate device memory
    float *d_M, *d_N, *d_P;
    cudaMalloc((void **)&d_M, r_M * c_M * sizeof(float));
    cudaMalloc((void **)&d_N, r_N * c_N * sizeof(float));
    cudaMalloc((void **)&d_P, r_P * c_P * sizeof(float));

    // Copy host matrices to device
    cudaMemcpy(d_M, M, r_M * c_M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, r_N * c_N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_P, 0, r_P * c_P * sizeof(float));

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    /*
     * cuBLAS uses column-major layout, so to simulate row-major C = A x B,
     * we compute: Cᵗ = Bᵗ x Aᵗ
     *
     * That is:
     *   C = A x B      (row-major)
     *   becomes
     *   Cᵗ = Bᵗ x Aᵗ   (column-major, cuBLAS-style)
     */

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                r_M, c_N, c_M, &alpha, d_M, r_M, d_N, r_N, &beta, d_P, r_P);

    // Copy result back to host
    float *result_T = (float *)malloc(r_P * c_P * sizeof(float));
    cudaMemcpy(result_T, d_P, r_P * c_P * sizeof(float), cudaMemcpyDeviceToHost);

    float *result = (float *)malloc(r_P * c_P * sizeof(float));
    transpose(result_T, result, c_P, r_P);

    // Validate result vs reference P
    int isValid = 1;
    for (int i = 0; i < r_P * c_P; i++)
    {
        if (fabs(result[i] - P[i]) > 1e-3f)
        {
            isValid = 0;
            printf("Mismatch at index %d: CUBLAS=%f, YOUR IMPLEMENTATION=%f\n", i, result[i], P[i]);
            break;
        }
    }

    // Cleanup
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(result);
    cublasDestroy(handle);

    return isValid;
}
