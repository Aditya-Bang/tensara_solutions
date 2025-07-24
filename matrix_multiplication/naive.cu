#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < K) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * K + j];
        }
        C[i * K + j] = sum;
    }
}

// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t n, size_t k) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_a, input_b, output_c, m, k, n);
    cudaDeviceSynchronize();
}
