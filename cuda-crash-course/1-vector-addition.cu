// module load cuda12.6/toolkit/12.6

#include <stdio.h>
#include <assert.h>

__global__ void vectorAdd(int* d_a, int* d_b, int *d_c, int N) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N) {
        d_c[tid] = d_a[tid] + d_b[tid];
    }
}

void matrix_init(int* m, int N) {
    for (int i = 0; i < N; i++) {
        m[i] = rand() % 100;
    }
}

void validate(int* h_a, int* h_b, int* h_c, int N) {
    for(int i = 0; i < N; i++) {
        assert(h_c[i] == h_a[i] + h_b[i]);
    }
    printf("Solution validated!\n");
}

int main() {
    int N = 1 << 16;

    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    size_t bytes = N * sizeof(int);

    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    matrix_init(h_a, N);
    matrix_init(h_b, N);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int NUM_THREADS = 256;
    int NUM_BLOCKS = (int)(N / NUM_THREADS);

    vectorAdd<<<NUM_THREADS, NUM_BLOCKS>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    validate(h_a, h_b, h_c, N);

    printf("Code executed successfully!\n");

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}