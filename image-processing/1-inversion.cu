#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <fstream>
#include <assert.h>

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

void write_to_stdout(void* context, void* data, int size) {
    fwrite(data, 1, size, stdout);
}

void save(unsigned char* data, int width, int height) {
    stbi_write_png_to_func(write_to_stdout, nullptr, width, height, 1, data, width);
}

int main()
{
    const char *filename = "dataset/lena_gray.gif";

    int width, height, channels;
    unsigned char *img = stbi_load(filename, &width, &height, &channels, 1);

    if (!img)
    {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return 1;
    }

    unsigned char *inverted_img = new unsigned char[width * height];
    invert(img, inverted_img, width, height);

    validate(img, inverted_img, width, height);

    save(inverted_img, width, height);

    delete[] inverted_img;
    stbi_image_free(img);

    return 0;
}
