#include <cuda_runtime.h>
#include <stdint.h>

__global__ void preprocess_kernel(
    const uint8_t *bgr,
    int source_width,
    int source_height,
    int source_stride,
    float *output,
    int output_width,
    int output_height,
    float scale,
    int resized_width,
    int resized_height,
    int pad_x,
    int pad_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output_width || y >= output_height) {
        return;
    }
    float red = 114.0f;
    float green = 114.0f;
    float blue = 114.0f;
    int local_x = x - pad_x;
    int local_y = y - pad_y;
    if (local_x >= 0 && local_x < resized_width &&
        local_y >= 0 && local_y < resized_height) {
        float source_x = (local_x + 0.5f) / scale - 0.5f;
        float source_y = (local_y + 0.5f) / scale - 0.5f;
        source_x = fminf(fmaxf(source_x, 0.0f), source_width - 1.0f);
        source_y = fminf(fmaxf(source_y, 0.0f), source_height - 1.0f);
        int x0 = static_cast<int>(floorf(source_x));
        int y0 = static_cast<int>(floorf(source_y));
        int x1 = min(x0 + 1, source_width - 1);
        int y1 = min(y0 + 1, source_height - 1);
        float wx = source_x - x0;
        float wy = source_y - y0;
        const uint8_t *p00 = bgr + y0 * source_stride + x0 * 3;
        const uint8_t *p01 = bgr + y0 * source_stride + x1 * 3;
        const uint8_t *p10 = bgr + y1 * source_stride + x0 * 3;
        const uint8_t *p11 = bgr + y1 * source_stride + x1 * 3;
        float w00 = (1.0f - wx) * (1.0f - wy);
        float w01 = wx * (1.0f - wy);
        float w10 = (1.0f - wx) * wy;
        float w11 = wx * wy;
        blue = p00[0] * w00 + p01[0] * w01 + p10[0] * w10 + p11[0] * w11;
        green = p00[1] * w00 + p01[1] * w01 + p10[1] * w10 + p11[1] * w11;
        red = p00[2] * w00 + p01[2] * w01 + p10[2] * w10 + p11[2] * w11;
    }
    int plane = output_width * output_height;
    int index = y * output_width + x;
    output[index] = red * (1.0f / 255.0f);
    output[plane + index] = green * (1.0f / 255.0f);
    output[2 * plane + index] = blue * (1.0f / 255.0f);
}

extern "C" int ball_launch_preprocess(
    const uint8_t *device_bgr,
    int source_width,
    int source_height,
    int source_stride,
    float *device_output,
    int output_width,
    int output_height,
    cudaStream_t stream) {
    if (!device_bgr || !device_output || source_width <= 0 || source_height <= 0 ||
        output_width <= 0 || output_height <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    float scale = fminf(
        static_cast<float>(output_width) / source_width,
        static_cast<float>(output_height) / source_height);
    int resized_width = static_cast<int>(roundf(source_width * scale));
    int resized_height = static_cast<int>(roundf(source_height * scale));
    int pad_x = (output_width - resized_width) / 2;
    int pad_y = (output_height - resized_height) / 2;
    dim3 block(16, 16);
    dim3 grid((output_width + block.x - 1) / block.x,
              (output_height + block.y - 1) / block.y);
    preprocess_kernel<<<grid, block, 0, stream>>>(
        device_bgr,
        source_width,
        source_height,
        source_stride,
        device_output,
        output_width,
        output_height,
        scale,
        resized_width,
        resized_height,
        pad_x,
        pad_y);
    return static_cast<int>(cudaGetLastError());
}
