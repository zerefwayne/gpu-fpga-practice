#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION

#include "../stb/stb_image.h"
#include "../stb/stb_image_write.h"

#include <iostream>
#include <fstream>
#include <assert.h>
#include <chrono>

size_t getImageSize(int width, int height, int channels)
{
    return static_cast<size_t>(width) * height * channels;
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
    unsigned char *img = stbi_load(filename, &width, &height, &channels, 1);

    if (!img)
    {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << width << "x" << height << "x" << 1 << std::endl;

    unsigned char *inverted_img = new unsigned char[width * height];

    auto start = std::chrono::high_resolution_clock::now();

    invert(img, inverted_img, width, height);

    auto stop = std::chrono::high_resolution_clock::now();

    validate(img, inverted_img, width, height);

    save("output.png", inverted_img, width, height);

    std::chrono::duration<double, std::milli> ms = stop - start;
    std::cout << "Inversion execution time: " << ms.count() << " ms\n";

    delete[] inverted_img;
    stbi_image_free(img);

    return 0;
}
