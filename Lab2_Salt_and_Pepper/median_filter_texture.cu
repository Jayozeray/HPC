#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;      \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BMPInfoHeader {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

static inline unsigned char rgbToGray(unsigned char r, unsigned char g, unsigned char b) {
    return static_cast<unsigned char>((77u * r + 150u * g + 29u * b) >> 8);
}

std::vector<unsigned char> loadBMPGray(const std::string& filename, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open input file: " + filename);
    }

    BMPFileHeader fileHeader{};
    BMPInfoHeader infoHeader{};

    file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
    file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

    if (!file) throw std::runtime_error("Failed to read BMP headers");
    if (fileHeader.bfType != 0x4D42) throw std::runtime_error("Input file is not a BMP");
    if (infoHeader.biSize < 40) throw std::runtime_error("Unsupported BMP header");
    if (infoHeader.biPlanes != 1) throw std::runtime_error("Unsupported BMP: biPlanes must be 1");
    if (infoHeader.biCompression != 0) throw std::runtime_error("Only uncompressed BMP is supported");
    if (infoHeader.biWidth <= 0 || infoHeader.biHeight == 0) throw std::runtime_error("Invalid BMP dimensions");

    if (infoHeader.biBitCount != 8 && infoHeader.biBitCount != 24 && infoHeader.biBitCount != 32) {
        throw std::runtime_error("Supported BMP formats: 8-bit, 24-bit, 32-bit");
    }

    if (infoHeader.biSize > sizeof(BMPInfoHeader)) {
        std::streamoff extra = static_cast<std::streamoff>(infoHeader.biSize) -
                               static_cast<std::streamoff>(sizeof(BMPInfoHeader));
        file.seekg(extra, std::ios::cur);
    }

    width = infoHeader.biWidth;
    height = (infoHeader.biHeight > 0) ? infoHeader.biHeight : -infoHeader.biHeight;
    const bool bottomUp = (infoHeader.biHeight > 0);

    std::vector<unsigned char> paletteGray;

    if (infoHeader.biBitCount == 8) {
        std::streamoff currentPos = static_cast<std::streamoff>(file.tellg());
        std::streamoff bytesBeforePixels = static_cast<std::streamoff>(fileHeader.bfOffBits) - currentPos;

        if (bytesBeforePixels > 0) {
            size_t paletteEntries = static_cast<size_t>(bytesBeforePixels / 4);
            paletteGray.resize(paletteEntries);

            for (size_t i = 0; i < paletteEntries; ++i) {
                unsigned char bgra[4];
                file.read(reinterpret_cast<char*>(bgra), 4);
                if (!file) throw std::runtime_error("Failed to read BMP palette");
                paletteGray[i] = rgbToGray(bgra[2], bgra[1], bgra[0]);
            }
        }
    }

    file.seekg(fileHeader.bfOffBits, std::ios::beg);

    const size_t rowStride =
        ((static_cast<size_t>(width) * infoHeader.biBitCount + 31u) / 32u) * 4u;

    std::vector<unsigned char> image(static_cast<size_t>(width) * height);
    std::vector<unsigned char> row(rowStride);

    for (int rowIdx = 0; rowIdx < height; ++rowIdx) {
        file.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(rowStride));
        if (!file) throw std::runtime_error("Failed to read BMP pixel data");

        int y = bottomUp ? (height - 1 - rowIdx) : rowIdx;

        if (infoHeader.biBitCount == 8) {
            for (int x = 0; x < width; ++x) {
                unsigned char idx = row[static_cast<size_t>(x)];
                unsigned char gray = idx;
                if (!paletteGray.empty() && static_cast<size_t>(idx) < paletteGray.size()) {
                    gray = paletteGray[idx];
                }
                image[static_cast<size_t>(y) * width + x] = gray;
            }
        } else if (infoHeader.biBitCount == 24) {
            for (int x = 0; x < width; ++x) {
                const size_t base = static_cast<size_t>(x) * 3u;
                unsigned char b = row[base + 0];
                unsigned char g = row[base + 1];
                unsigned char r = row[base + 2];
                image[static_cast<size_t>(y) * width + x] = rgbToGray(r, g, b);
            }
        } else {
            for (int x = 0; x < width; ++x) {
                const size_t base = static_cast<size_t>(x) * 4u;
                unsigned char b = row[base + 0];
                unsigned char g = row[base + 1];
                unsigned char r = row[base + 2];
                image[static_cast<size_t>(y) * width + x] = rgbToGray(r, g, b);
            }
        }
    }

    return image;
}

void saveBMPGray(const std::string& filename,
                 const std::vector<unsigned char>& image,
                 int width,
                 int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open output file: " + filename);

    const size_t rowStride = ((static_cast<size_t>(width) + 3u) / 4u) * 4u;
    const uint32_t paletteSize = 256u * 4u;
    const uint32_t pixelDataSize = static_cast<uint32_t>(rowStride * height);

    BMPFileHeader fileHeader{};
    fileHeader.bfType = 0x4D42;
    fileHeader.bfOffBits = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + paletteSize;
    fileHeader.bfSize = fileHeader.bfOffBits + pixelDataSize;

    BMPInfoHeader infoHeader{};
    infoHeader.biSize = sizeof(BMPInfoHeader);
    infoHeader.biWidth = width;
    infoHeader.biHeight = height;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 8;
    infoHeader.biCompression = 0;
    infoHeader.biSizeImage = pixelDataSize;
    infoHeader.biClrUsed = 256;
    infoHeader.biClrImportant = 256;

    file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));

    for (int i = 0; i < 256; ++i) {
        unsigned char entry[4] = {
            static_cast<unsigned char>(i),
            static_cast<unsigned char>(i),
            static_cast<unsigned char>(i),
            0
        };
        file.write(reinterpret_cast<const char*>(entry), 4);
    }

    std::vector<unsigned char> row(rowStride, 0);

    for (int y = height - 1; y >= 0; --y) {
        std::memcpy(row.data(), &image[static_cast<size_t>(y) * width], static_cast<size_t>(width));
        file.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(rowStride));
    }

    if (!file) throw std::runtime_error("Failed to write output BMP");
}

__device__ __forceinline__ void bubbleSort9(unsigned char* v) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8 - i; ++j) {
            if (v[j] > v[j + 1]) {
                unsigned char t = v[j];
                v[j] = v[j + 1];
                v[j + 1] = t;
            }
        }
    }
}

__global__ void median9TextureKernel(cudaTextureObject_t texObj,
                                     unsigned char* output,
                                     int width,
                                     int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    unsigned char window[9];
    int k = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            window[k++] = tex2D<unsigned char>(
                texObj,
                static_cast<float>(x + dx) + 0.5f,
                static_cast<float>(y + dy) + 0.5f
            );
        }
    }

    bubbleSort9(window);
    output[static_cast<size_t>(y) * width + x] = window[4];
}

int main(int argc, char** argv) {
    using clock_type = std::chrono::steady_clock;

    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " input.bmp output.bmp" << std::endl;
            return EXIT_FAILURE;
        }

        const std::string inputFile = argv[1];
        const std::string outputFile = argv[2];

        auto total_start = clock_type::now();

        int width = 0, height = 0;
        std::vector<unsigned char> inputImage = loadBMPGray(inputFile, width, height);
        std::vector<unsigned char> outputImage(static_cast<size_t>(width) * height);

        int device = 0;
        CUDA_CHECK(cudaSetDevice(device));

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        std::cout << "GPU: " << prop.name << std::endl;
        std::cout << "Image size: " << width << " x " << height << std::endl;

        auto gpu_stage_start = clock_type::now();

        unsigned char* d_input = nullptr;
        unsigned char* d_output = nullptr;
        size_t pitch = 0;

        CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_input),
                                   &pitch,
                                   static_cast<size_t>(width) * sizeof(unsigned char),
                                   static_cast<size_t>(height)));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output),
                              static_cast<size_t>(width) * height * sizeof(unsigned char)));

        CUDA_CHECK(cudaMemcpy2D(d_input,
                                pitch,
                                inputImage.data(),
                                static_cast<size_t>(width) * sizeof(unsigned char),
                                static_cast<size_t>(width) * sizeof(unsigned char),
                                static_cast<size_t>(height),
                                cudaMemcpyHostToDevice));

        cudaResourceDesc resDesc{};
        std::memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = d_input;
        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<unsigned char>();
        resDesc.res.pitch2D.width = static_cast<size_t>(width);
        resDesc.res.pitch2D.height = static_cast<size_t>(height);
        resDesc.res.pitch2D.pitchInBytes = pitch;

        cudaTextureDesc texDesc{};
        std::memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaTextureObject_t texObj = 0;
        CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

        median9TextureKernel<<<grid, block>>>(texObj, d_output, width, height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, 0));
        median9TextureKernel<<<grid, block>>>(texObj, d_output, width, height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

        CUDA_CHECK(cudaMemcpy(outputImage.data(),
                              d_output,
                              static_cast<size_t>(width) * height * sizeof(unsigned char),
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaDeviceSynchronize());
        auto gpu_stage_stop = clock_type::now();

        saveBMPGray(outputFile, outputImage, width, height);
        auto total_stop = clock_type::now();

        double gpu_stage_ms =
            std::chrono::duration<double, std::milli>(gpu_stage_stop - gpu_stage_start).count();

        double total_ms =
            std::chrono::duration<double, std::milli>(total_stop - total_start).count();

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Kernel time (CUDA events): " << kernel_ms << " ms" << std::endl;
        std::cout << "GPU stage time (alloc + H2D + texture + kernel + D2H): "
                  << gpu_stage_ms << " ms" << std::endl;
        std::cout << "End-to-end time (read BMP + GPU stage + write BMP): "
                  << total_ms << " ms" << std::endl;
        std::cout << "Saved result to: " << outputFile << std::endl;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaDestroyTextureObject(texObj));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));

        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
