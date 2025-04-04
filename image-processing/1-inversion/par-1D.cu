// Inversion execution time: 0.084768 ms

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION

#include "../stb/stb_image.h"
#include "../stb/stb_image_write.h"

#include <iostream>
#include <fstream>
#include <assert.h>
#include <cuda_runtime.h>

__global__ void invertGpu(unsigned char *d_input, unsigned char *d_output, int width, int height) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < width * height) {
        d_output[tid] = 255 - d_input[tid];
    }
}

void invert(unsigned char *input, unsigned char *output, int width, int height) {
    int size = width * height;

    for (int i = 0; i < size; i++) {
        output[i] = 255 - input[i];
    }
}

void validate(unsigned char *input, unsigned char *output, int width, int height) {
    int size = width * height;
    for(int i = 0; i < size; i++) {
        assert(output[i] == (255 - input[i]));
    }
}

bool save(const char* filename, unsigned char* data, int width, int height) {
    int success = stbi_write_png(filename, width, height, 1, data, width);
    return success != 0;
}

int main()
{
    const char *filename = "input.gif";

    int width, height, channels;
    unsigned char *h_input = stbi_load(filename, &width, &height, &channels, 1);

    if (!h_input)
    {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << width << "x" << height << "x" << 1 << std::endl;

    // Image loaded

    size_t bytes = width * height * sizeof(unsigned char);

    unsigned char *h_output;
    h_output = (unsigned char*)malloc(bytes);

    unsigned char *d_input, *d_output;
    
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    int NUM_THREADS = 256;
    int NUM_BLOCKS = (int)((width * height)/NUM_THREADS);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    invertGpu<<<NUM_BLOCKS, NUM_THREADS>>>(d_input, d_output, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    validate(h_output, h_input, width, height);

    save("output.png", h_output, width, height);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    delete[] h_output;
    stbi_image_free(h_input);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
