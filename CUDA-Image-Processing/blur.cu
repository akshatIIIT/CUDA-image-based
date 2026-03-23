#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

__global__ void blurKernel(unsigned char* input, unsigned char* output,
                           int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; c++) {
        int sum = 0;
        int count = 0;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int nx = x + kx;
                int ny = y + ky;

                if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                    sum += input[(ny * width + nx) * channels + c];
                    count++;
                }
            }
        }

        output[(y * width + x) * channels + c] = sum / count;
    }
}

int main() {
    std::string inputDir = "images/input/";
    std::string outputDir = "images/output/";

    fs::create_directories(outputDir);

    int count = 0;

    for (auto& p : fs::directory_iterator(inputDir)) {
        std::string inputPath = p.path().string();
        std::string filename = p.path().filename().string();
        std::string outputPath = outputDir + filename;

        cv::Mat img = cv::imread(inputPath);

        if (img.empty()) {
            std::cout << "Failed to load: " << inputPath << std::endl;
            continue;
        }

        int width = img.cols;
        int height = img.rows;
        int channels = img.channels();
        int size = width * height * channels * sizeof(unsigned char);

        unsigned char *d_in, *d_out;
        cudaMalloc(&d_in, size);
        cudaMalloc(&d_out, size);

        cudaMemcpy(d_in, img.data, size, cudaMemcpyHostToDevice);

        dim3 block(16,16);
        dim3 grid((width+15)/16, (height+15)/16);

        blurKernel<<<grid, block>>>(d_in, d_out, width, height, channels);

        cudaMemcpy(img.data, d_out, size, cudaMemcpyDeviceToHost);

        cv::imwrite(outputPath, img);

        cudaFree(d_in);
        cudaFree(d_out);

        count++;
        std::cout << "Processed: " << filename << std::endl;
    }

    std::cout << "Total images processed: " << count << std::endl;
}
