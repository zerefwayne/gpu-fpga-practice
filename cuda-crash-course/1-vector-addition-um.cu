// module load cuda12.6/toolkit/12.6

#include <stdio.h>
#include <assert.h>

__global__ void vectorAddUM(int* d_a, int* d_b, int *d_c, int N) {
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
    int id = cudaGetDevice(&id);
    int N = 1 << 16;

    int *a, *b, *c;

    size_t bytes = N * sizeof(int);

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    matrix_init(a, N);
    matrix_init(b, N);

    int NUM_THREADS = 256;
    int NUM_BLOCKS = (int)(N / NUM_THREADS);

    cudaMemPrefetchAsync(&a, bytes, id);
    cudaMemPrefetchAsync(&b, bytes, id);

    vectorAddUM<<<NUM_THREADS, NUM_BLOCKS>>>(a, b, c, N);
    
    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    validate(a, b, c, N);

    printf("Code executed successfully!\n");

    // Free memory on device
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}