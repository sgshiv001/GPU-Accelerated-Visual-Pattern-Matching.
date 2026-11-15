#pragma once

// Row-major matrices: A[M,K], B[K,N], and C[M,N].
__global__ void gemm_naive(const float* a, const float* b, float* c, int m, int n, int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (int p = 0; p < k; ++p) sum = fmaf(a[row * k + p], b[p * n + col], sum);
    c[row * n + col] = sum;
}

constexpr int tile = 16;

__global__ void gemm_shared(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float as[tile][tile];
    __shared__ float bs[tile][tile];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int row = blockIdx.y * tile + y;
    const int col = blockIdx.x * tile + x;

    float sum = 0.0f;
    for (int base = 0; base < k; base += tile) {
        as[y][x] = (row < m && base + x < k) ? a[row * k + base + x] : 0.0f;
        bs[y][x] = (base + y < k && col < n) ? b[(base + y) * n + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int p = 0; p < tile; ++p) sum = fmaf(as[y][p], bs[p][x], sum);
        __syncthreads();
    }
    if (row < m && col < n) c[row * n + col] = sum;
}

constexpr int block_x = 32;
constexpr int block_y = 8;
constexpr int rows_per_thread = 8;
constexpr int cols_per_thread = 2;
constexpr int block_m = block_y * rows_per_thread;
constexpr int block_n = block_x * cols_per_thread;
constexpr int block_k = 16;

// Each warp owns one output row at a time, with adjacent lanes storing adjacent columns.
__global__ void gemm_register(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float as[block_m][block_k];
    __shared__ float bs[block_k][block_n];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int lane_in_block = y * block_x + x;
    const int row0 = blockIdx.y * block_m;
    const int col0 = blockIdx.x * block_n;
    float acc[rows_per_thread][cols_per_thread] = {};

    for (int base = 0; base < k; base += block_k) {
        #pragma unroll
        for (int i = lane_in_block; i < block_m * block_k; i += block_x * block_y) {
            const int row = row0 + i / block_k;
            const int p = base + i % block_k;
            as[i / block_k][i % block_k] = (row < m && p < k) ? a[row * k + p] : 0.0f;
        }
        #pragma unroll
        for (int i = lane_in_block; i < block_k * block_n; i += block_x * block_y) {
            const int p = base + i / block_n;
            const int col = col0 + i % block_n;
            bs[i / block_n][i % block_n] = (p < k && col < n) ? b[p * n + col] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int p = 0; p < block_k; ++p) {
            const float b0 = bs[p][x];
            const float b1 = bs[p][x + block_x];
            #pragma unroll
            for (int i = 0; i < rows_per_thread; ++i) {
                const float av = as[y + i * block_y][p];
                acc[i][0] = fmaf(av, b0, acc[i][0]);
                acc[i][1] = fmaf(av, b1, acc[i][1]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < rows_per_thread; ++i) {
        const int row = row0 + y + i * block_y;
        #pragma unroll
        for (int j = 0; j < cols_per_thread; ++j) {
            const int col = col0 + x + j * block_x;
            if (row < m && col < n) c[row * n + col] = acc[i][j];
        }
    }
}
