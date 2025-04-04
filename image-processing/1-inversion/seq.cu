#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION

#include "../stb/stb_image.h"
#include "../stb/stb_image_write.h"

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

    // Print a few pixels
    for (int i = 0; i < 10; i++)
    {
        std::cout << "Pixel " << i << ": " << (int)img[i] << std::endl;
    }

    unsigned char *inverted_img = new unsigned char[width * height];
    invert(img, inverted_img, width, height);

    std::cout << std::endl;

    // Print a few pixels for inverted image
    for (int i = 0; i < 10; i++)
    {
        std::cout << "Pixel " << i << ": " << (int)inverted_img[i] << std::endl;
    }

    validate(img, inverted_img, width, height);

    save("output.png", inverted_img, width, height);

    delete[] inverted_img;
    stbi_image_free(img);

    return 0;
}
