#include <stdio.h>
#include <assert.h>

__global__ void matrixMul(int *d_a, int *d_b, int *d_c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int res = 0;

    if (row < N && col < N) {
        for(int k = 0; k < N; k++) {
            res += d_a[row * N + k] + d_b[k * N + col];
        }
    }

    d_c[row*N + col] = res;
}

void init_matrix(int *a, int N) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            a[i*N + j] = rand() % 100;
        }
    }
}

void validate(int *h_a, int *h_b, int *h_c, int N) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            int res = 0;
            for(int k = 0; k < N; k++) {
                res += h_a[i*N + k] + h_b[k*N + j];
            }
            assert(h_c[i*N + j] == res);
        }
    }
    printf("Solution validated!\n");
}

int main() {
    int N = 1 << 10;
    size_t bytes = N * N * sizeof(int);

    int *h_a, *h_b, *h_c;

    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    int *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    init_matrix(h_a, N);
    init_matrix(h_b, N);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToHost);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int NUM_THREADS = 16;
    int NUM_BLOCKS = (int)ceil(N/NUM_THREADS);

    dim3 grid(NUM_BLOCKS, NUM_BLOCKS);
    dim3 threads(NUM_THREADS, NUM_THREADS);

    matrixMul<<<grid, threads>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    validate(h_a, h_b, h_c, N);

    printf("Program executed successfully!\n");

    return 0;
}