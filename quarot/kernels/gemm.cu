#include <cstdint>
#include <cutlass/gemm/device/gemm.h>
#include <gemm.h>

#include "../../../src/hadamard.cuh"

namespace {

struct MatMulParams {
    constexpr static size_t N_THR = 256;

    constexpr static size_t SHM_M = 128;
    constexpr static size_t SHM_N = 256;
    constexpr static size_t SHM_K = 128;

    constexpr static size_t CPY_K = 32; // Copy at the granularity of 16 bytes

    constexpr static size_t MMA_M = 16;
    constexpr static size_t MMA_N = 8;
    constexpr static size_t MMA_K = 64;

    constexpr static size_t REG_M = 4;
    constexpr static size_t REG_N = 8;
};

template <typename A, typename B> auto ceil_div(A a, B b) { return (a + b - 1) / b; }

// For ldmatrix, a K-stride (32x+16) bytes leads to bank-conflict free access.
constexpr size_t round_up_to_32x_plus_16(size_t x) { return (((x - 16) + 31) & ~31) + 16; }

// template <typename P>
// constexpr size_t SHM_K_STRIDE = [] {
//     constexpr size_t EPB = 2; // Elements per byte
//     static_assert(P::SHM_K % (EPB * sizeof(int32_t)) == 0);
//     constexpr size_t SHM_K_INT32S = P::SHM_K / (EPB * sizeof(int32_t));
//     static_assert(SHM_K_INT32S >= 4);
//     return (((SHM_K_INT32S - 4 + 7) & ~7) + 4) * sizeof(int32_t);
// }();

// template <typename P> constexpr size_t SHM_A_SIZE = P::SHM_M * SHM_K_STRIDE<P>;
// template <typename P> constexpr size_t SHM_B_SIZE = P::SHM_N * SHM_K_STRIDE<P>;
// template <typename P> constexpr size_t SHM_SIZE = 2 * (SHM_A_SIZE<P> + SHM_B_SIZE<P>);

// template <typename P>
// constexpr size_t SHM_SIZE_FUSED = 2 * (SHM_A_SIZE<P> + SHM_B_SIZE<P> + P::SHM_M * sizeof(half));

template <typename T> __device__ void async_copy(T *dst, const T *src) {
    static_assert(sizeof(T) == 16 || sizeof(T) == 8 || sizeof(T) == 4);
    const uint32_t dst_smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    if constexpr (sizeof(T) == 16) {
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" ::"r"(dst_smem), "l"(src));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst_smem), "l"(src));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst_smem), "l"(src));
    }
}

__device__ void async_copy_waitall() { asm volatile("cp.async.wait_all;\n" ::); }

template <typename T> __device__ void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <typename P> struct MatmulDerivedParams {
    constexpr static size_t SHM_K_STRIDE = round_up_to_32x_plus_16(P::SHM_K / 2);
    constexpr static size_t SHM_A_SIZE = P::SHM_M * SHM_K_STRIDE;
    constexpr static size_t SHM_B_SIZE = P::SHM_N * SHM_K_STRIDE;
    constexpr static size_t SHM_SIZE = 2 * (SHM_A_SIZE + SHM_B_SIZE);
};

template <typename P>
__global__ __launch_bounds__(P::N_THR) void matmul_kernel(const uint8_t *A, const uint8_t *B, int32_t *C,
                                                          uint32_t M, uint32_t N, uint32_t K) {
    using D = MatmulDerivedParams<P>;
    constexpr size_t EPB = 2; // Elements per byte

    static_assert(P::N_THR % 32 == 0, "N_THR must be a multiple of 32");

    static_assert(P::SHM_K % P::CPY_K == 0, "SHM_K must be a multiple of CPY_K");
    // Either partial copy or full tiled copy
    static_assert(P::N_THR * P::CPY_K >= P::SHM_M * P::SHM_K ||
                      P::SHM_M * P::SHM_K % (P::N_THR * P::CPY_K) == 0,
                  "N_THR * CPY_K must be >= SHM_M * SHM_K");
    static_assert(P::N_THR * P::CPY_K >= P::SHM_N * P::SHM_K ||
                      P::SHM_N * P::SHM_K % (P::N_THR * P::CPY_K) == 0,
                  "N_THR * CPY_K must be >= SHM_N * SHM_K");

    // clang-format off
    using CopyType = std::conditional_t<P::CPY_K ==  4 * EPB, uint32_t,
                     std::conditional_t<P::CPY_K ==  8 * EPB, uint64_t,
                     std::conditional_t<P::CPY_K == 16 * EPB, float4, void>>>;
    // clang-format on

    constexpr size_t N_WRP = P::N_THR / 32;
    static_assert(P::SHM_M % (P::REG_M * P::MMA_M) == 0 && P::SHM_N % (P::REG_N * P::MMA_N) == 0, "");
    // Number of warp tiles along M and N
    constexpr size_t WRP_M = P::SHM_M / (P::REG_M * P::MMA_M);
    constexpr size_t WRP_N = P::SHM_N / (P::REG_N * P::MMA_N);
    static_assert(WRP_M * WRP_N == N_WRP, "");

    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t *__restrict__ smem_A = smem;
    uint8_t *__restrict__ smem_B = smem + D::SHM_A_SIZE * 2;

    const size_t tid = threadIdx.x;
    const size_t wid = tid / 32;
    const size_t lid = tid % 32;
    const size_t gid = lid / 8;

    uint8_t *__restrict__ smem_A_cur = smem_A, *__restrict__ smem_A_next = smem_A + D::SHM_A_SIZE;
    uint8_t *__restrict__ smem_B_cur = smem_B, *__restrict__ smem_B_next = smem_B + D::SHM_B_SIZE;

#define unroll _Pragma("unroll")
    const auto async_copy_to_smem = [&](size_t k) {
        unroll for (size_t i = 0; i < P::SHM_M * P::SHM_K; i += P::N_THR * P::CPY_K) {
            // Copy to smem_A
            const size_t tile_idx = i + tid * P::CPY_K;
            if constexpr (P::N_THR * P::CPY_K > P::SHM_M * P::SHM_K) {
                if (tile_idx >= P::SHM_M * P::SHM_K) {
                    break;
                }
            }
            const size_t tile_m = tile_idx / P::SHM_K;
            const size_t gmem_m = tile_m + blockIdx.x * P::SHM_M;
            const size_t tile_k = tile_idx % P::SHM_K;
            const size_t gmem_k = k + tile_k;

            const CopyType *src = reinterpret_cast<const CopyType *>(A + gmem_m * (K / EPB) + gmem_k / EPB);
            CopyType *dst =
                reinterpret_cast<CopyType *>(smem_A_next + tile_m * D::SHM_K_STRIDE + tile_k / EPB);
            async_copy(dst, src);
        }
        unroll for (size_t i = 0; i < P::SHM_N * P::SHM_K; i += P::N_THR * P::CPY_K) {
            // Copy to smem_B
            const size_t tile_idx = i + tid * P::CPY_K;
            if constexpr (P::N_THR * P::CPY_K > P::SHM_N * P::SHM_K) {
                if (tile_idx >= P::SHM_N * P::SHM_K) {
                    break;
                }
            }
            const size_t tile_n = tile_idx / P::SHM_K;
            const size_t gmem_n = tile_n + blockIdx.y * P::SHM_N;
            const size_t tile_k = tile_idx % P::SHM_K;
            const size_t gmem_k = k + tile_k;
            const CopyType *src = reinterpret_cast<const CopyType *>(B + gmem_n * (K / EPB) + gmem_k / EPB);
            CopyType *dst =
                reinterpret_cast<CopyType *>(smem_B_next + tile_n * D::SHM_K_STRIDE + tile_k / EPB);
            async_copy(dst, src);
        }
    };

    int32_t c[P::REG_M][P::REG_N][4] = {0};
    const size_t mma_wrp_m = wid / WRP_N * P::REG_M * P::MMA_M;
    const size_t mma_wrp_n = wid % WRP_N * P::REG_N * P::MMA_N;
    const size_t mma_trd_m_ld_base = mma_wrp_m + lid % 8 + (gid % 2) * 8;
    const size_t mma_trd_n_ld_base = mma_wrp_n + lid % 8;

    async_copy_to_smem(0);

    for (size_t k = 0; k < K; k += P::SHM_K) {
        swap(smem_A_cur, smem_A_next);
        swap(smem_B_cur, smem_B_next);
        async_copy_waitall();
        __syncthreads();
        if (k + P::SHM_K < K) {
            async_copy_to_smem(k + P::SHM_K);
        }
        unroll for (size_t k_ = 0; k_ < P::SHM_K; k_ += P::MMA_K) {
            int32_t a[P::REG_M][4], b[P::REG_N][2];
            unroll for (size_t i = 0; i < P::REG_M; i++) {
                const size_t mma_trd_m = mma_trd_m_ld_base + i * P::MMA_M;
                const size_t mma_trd_k = k_ + (gid / 2) * 32;
                const size_t addr =
                    __cvta_generic_to_shared(smem_A_cur + mma_trd_m * D::SHM_K_STRIDE + mma_trd_k / EPB);
                asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                    : "=r"(a[i][0]), "=r"(a[i][1]), "=r"(a[i][2]), "=r"(a[i][3])
                    : "l"(addr));
            }
            unroll for (size_t j = 0; j < P::REG_N; j++) {
                const size_t mma_trd_n = mma_trd_n_ld_base + j * P::MMA_N;
                const size_t mma_trd_k = k_ + gid * 32;
                const size_t addr =
                    __cvta_generic_to_shared(smem_B_cur + mma_trd_n * D::SHM_K_STRIDE + mma_trd_k / EPB);
                asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                    : "=r"(b[j][0]), "=r"(b[j][1])
                    : "l"(addr));
            }
            unroll for (size_t i = 0; i < P::REG_M; i++) {
                unroll for (size_t j = 0; j < P::REG_N; j++) {
                    asm("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32"
                        " {%0, %1, %2, %3},"
                        " {%4, %5, %6, %7},"
                        " {%8, %9},"
                        " {%10, %11, %12, %13};"
                        : "=r"(c[i][j][0]), "=r"(c[i][j][1]), "=r"(c[i][j][2]), "=r"(c[i][j][3])
                        : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]), "r"(b[j][0]), "r"(b[j][1]),
                          "r"(c[i][j][0]), "r"(c[i][j][1]), "r"(c[i][j][2]), "r"(c[i][j][3]));
                }
            }
        }
    }
    unroll for (size_t i = 0; i < P::REG_M; i++) {
        unroll for (size_t j = 0; j < P::REG_N; j++) {
            const size_t mma_trd_m = mma_wrp_m + i * P::MMA_M + lid / 4;
            const size_t mma_trd_n = mma_wrp_n + j * P::MMA_N + (lid % 4) * 2;
            const size_t gmem_m = mma_trd_m + blockIdx.x * P::SHM_M;
            const size_t gmem_n = mma_trd_n + blockIdx.y * P::SHM_N;
            assert(gmem_m < M && gmem_n < N);
            C[(gmem_m + 0) * N + gmem_n + 0] = c[i][j][0];
            C[(gmem_m + 0) * N + gmem_n + 1] = c[i][j][1];
            C[(gmem_m + 8) * N + gmem_n + 0] = c[i][j][2];
            C[(gmem_m + 8) * N + gmem_n + 1] = c[i][j][3];
        }
    }
#undef unroll
}

struct MatMulHadamardParams {
    constexpr static size_t N_THR = 256;

    constexpr static size_t SHM_M = 16;
    constexpr static size_t SHM_N = 512;
    constexpr static size_t SHM_K = 128;

    // constexpr static size_t CPY_K = 32; // Copy at the granularity of 16 bytes

    constexpr static size_t MMA_M = 8;
    constexpr static size_t MMA_N = 16;
    constexpr static size_t MMA_K = 64;

    constexpr static size_t REG_M = 2;
    constexpr static size_t REG_N = 4;
};

template <typename P> struct MatmulHadamardDerivedParams {
    // K-stride for A (loaded to shmem in half precision)
    constexpr static size_t SHM_A_K_STRIDE = round_up_to_32x_plus_16(P::SHM_K * sizeof(half));
    // K-stride for B (loaded to shmem in 4-bit quantized format)
    constexpr static size_t SHM_B_K_STRIDE = round_up_to_32x_plus_16(P::SHM_K / 2);
    // Size of shared memory for A and B
    constexpr static size_t SHM_A_SIZE = P::SHM_M * SHM_A_K_STRIDE;
    constexpr static size_t SHM_B_SIZE = P::SHM_N * SHM_B_K_STRIDE;
    // Size for the scale factors (one per group)
    constexpr static size_t SHM_A_SCALE_SIZE = P::SHM_M * sizeof(half);
    // *2 for double buffering
    constexpr static size_t SHM_SIZE = 2 * (SHM_A_SIZE + SHM_B_SIZE + SHM_A_SCALE_SIZE);
};

/// Fused hadamard-linear kernel with group quantization.
///
/// Conceptually: fp16 input -> Hadamard -> 16-to-4 bit quantization -> int4 linear -> 32-to-16 quantization
template <typename P>
__global__ __launch_bounds__(P::N_THR) void matmul_hadamard_kernel(const half *A, const uint8_t *B, half *C,
                                                                   uint32_t M, uint32_t N, uint32_t K) {
    using D = MatmulHadamardDerivedParams<P>;

    constexpr size_t EPB = 2; // Elements per byte

    static_assert(P::N_THR % 32 == 0, "N_THR must be a multiple of 32");

    constexpr size_t N_WRP = P::N_THR / 32;
    static_assert(P::SHM_M % (P::REG_M * P::MMA_M) == 0 && P::SHM_N % (P::REG_N * P::MMA_N) == 0, "");
    // Number of warp tiles along M and N
    constexpr size_t WRP_M = P::SHM_M / (P::REG_M * P::MMA_M);
    constexpr size_t WRP_N = P::SHM_N / (P::REG_N * P::MMA_N);
    static_assert(WRP_M * WRP_N == N_WRP, "");

    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t *__restrict__ smem_A = smem;
    uint8_t *__restrict__ smem_B = smem + D::SHM_A_SIZE * 2;
    half *__restrict__ smem_A_scale = reinterpret_cast<half *>(smem_B + D::SHM_B_SIZE * 2);

    const size_t tid = threadIdx.x;
    const size_t wid = tid / 32;
    const size_t lid = tid % 32;
    const size_t gid = lid / 8;

    uint8_t *__restrict__ smem_A_cur = smem_A, *__restrict__ smem_A_next = smem_A + D::SHM_A_SIZE;
    uint8_t *__restrict__ smem_B_cur = smem_B, *__restrict__ smem_B_next = smem_B + D::SHM_B_SIZE;

    const size_t gmem_m_base = blockIdx.x * P::SHM_M;
    const size_t gmem_n_base = blockIdx.y * P::SHM_N;

#define unroll _Pragma("unroll")

    const auto async_copy_to_smem = [&](size_t k) {
        static_assert(P::SHM_K == 128 || P::SHM_K == 256);
        { // Copy to smem_A
            static_assert(P::SHM_M % N_WRP == 0);
            using CopyType = std::conditional_t<P::SHM_K == 128, float2, float4>;
            constexpr size_t EPC = sizeof(CopyType) / sizeof(half);

            const size_t tile_m_base = wid;
            const size_t tile_k = lid * EPC;
            const size_t gmem_k = k + tile_k;

            unroll for (size_t i = 0; i < P::SHM_M; i += N_WRP) {
                const size_t tile_m = tile_m_base + i;
                const size_t gmem_m = tile_m + gmem_m_base;
                const CopyType *src = reinterpret_cast<const CopyType *>(A + gmem_m * K + gmem_k);
                CopyType *dst = reinterpret_cast<CopyType *>(smem_A_next + tile_m * D::SHM_A_K_STRIDE +
                                                             tile_k * sizeof(half));
                async_copy(dst, src);
            }
        }
        { // Copy to smem_B
            // Copy at the granularity of 16 bytes. How many 16 bytes are there in a row?
            constexpr size_t T = P::SHM_K / 32; // Every 16 bytes hold 32 int4s
            static_assert(P::SHM_N % (N_WRP / T * 32) == 0);
            const size_t gid = lid / T;
            const size_t tile_n_base =
                T * /*(T == 8 ? gid : (gid / 2 + 4 * (gid % 2)))*/ gid + (wid % T) + (wid / T) * 32;
            const size_t tile_k = (lid % T) * 32;
            const size_t gmem_k = k + tile_k;
            unroll for (size_t i = 0; i < P::SHM_N; i += (N_WRP / T) * 32) {
                const size_t tile_n = tile_n_base + i;
                const size_t gmem_n = tile_n + gmem_n_base;
                const float4 *src = reinterpret_cast<const float4 *>(B + gmem_n * (K / EPB) + gmem_k / EPB);
                float4 *dst =
                    reinterpret_cast<float4 *>(smem_B_next + tile_n * D::SHM_B_K_STRIDE + tile_k / EPB);
                async_copy(dst, src);
            }
        }
    };

    float cf[P::REG_M][P::REG_N][4] = {0};
    int32_t c[P::REG_M][P::REG_N][4] = {0};

    const size_t mma_wrp_m = wid / WRP_N * P::REG_M * P::MMA_M;
    const size_t mma_wrp_n = wid % WRP_N * P::REG_N * P::MMA_N;
    const size_t mma_trd_m_ld_base = mma_wrp_m + lid % 8;
    const size_t mma_trd_n_ld_base = mma_wrp_n + lid % 8 + (gid % 2) * 8;

    async_copy_to_smem(0);

    for (size_t k = 0; k < K; k += P::SHM_K) {
        swap(smem_A_cur, smem_A_next);
        swap(smem_B_cur, smem_B_next);
        // swap(smem_A_scale_cur, smem_A_scale_next);
        async_copy_waitall();

        __syncthreads();
        if (k + P::SHM_K < K) {
            async_copy_to_smem(k + P::SHM_K);
        }

        // Hadamard to A in-place
        unroll for (size_t i = 0; i < P::SHM_M; i += N_WRP) {
            const size_t tile_m = i + wid;
            hadamard_transform_group_quantize<P::SHM_K, 32>(smem_A_cur + tile_m * D::SHM_A_K_STRIDE,
                                                            smem_A_scale + tile_m);
        }
        __syncthreads();
        // Zero out the accumulation buffer
        unroll for (size_t i = 0; i < P::REG_M; i++) {
            unroll for (size_t j = 0; j < P::REG_N; j++) {
                unroll for (size_t k_ = 0; k_ < 4; k_++) { c[i][j][k_] = 0; }
            }
        }
        // Transposed matrix multiplication: the MMA instruction expects the "A" matrix to be 16x64 (mxk) and
        // "B" to be 8x64 (nxk), both k-major. In our case, it is better to load the real A matrix into the B
        // position of the instruction and the real B matrix into the A position. This is fine because they
        // have compatible layouts. This way, the tiles we get are shorter in the M dimension and longer in
        // the N dimention. This has the following advantages:
        // 1. The number of hadamard transforms we need to do is porportional to the M size of the tile, so
        // the smaller the M, the better.
        // 2. Having longer N dimension means that we'll have fewer thread blocks using the same M range but
        // different N ranges. This improves the work efficiency of our algorithm.
        // Of course, this would mean we have transposed output.
        unroll for (size_t k_ = 0; k_ < P::SHM_K; k_ += P::MMA_K) {
            int32_t b[P::REG_N][4], a[P::REG_M][2];
            unroll for (size_t j = 0; j < P::REG_N; j++) {
                const size_t mma_trd_n = mma_trd_n_ld_base + j * P::MMA_N;
                const size_t mma_trd_k = k_ + (gid / 2) * 32;
                const size_t addr =
                    __cvta_generic_to_shared(smem_B_cur + mma_trd_n * D::SHM_B_K_STRIDE + mma_trd_k / EPB);
                asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                    : "=r"(b[j][0]), "=r"(b[j][1]), "=r"(b[j][2]), "=r"(b[j][3])
                    : "l"(addr));
            }
            unroll for (size_t i = 0; i < P::REG_M; i++) {
                const size_t mma_trd_m = mma_trd_m_ld_base + i * P::MMA_M;
                const size_t mma_trd_k = k_ + gid * 32;
                const size_t addr =
                    __cvta_generic_to_shared(smem_A_cur + mma_trd_m * D::SHM_A_K_STRIDE + mma_trd_k / EPB);
                asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                    : "=r"(a[i][0]), "=r"(a[i][1])
                    : "l"(addr));
            }
            unroll for (size_t i = 0; i < P::REG_M; i++) {
                unroll for (size_t j = 0; j < P::REG_N; j++) {
                    asm("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32"
                        " {%0, %1, %2, %3},"
                        " {%4, %5, %6, %7},"
                        " {%8, %9},"
                        " {%10, %11, %12, %13};"
                        : "=r"(c[i][j][0]), "=r"(c[i][j][1]), "=r"(c[i][j][2]), "=r"(c[i][j][3])
                        : "r"(b[j][0]), "r"(b[j][1]), "r"(b[j][2]), "r"(b[j][3]), "r"(a[i][0]), "r"(a[i][1]),
                          "r"(c[i][j][0]), "r"(c[i][j][1]), "r"(c[i][j][2]), "r"(c[i][j][3]));
                }
            }
        }
        unroll for (size_t i = 0; i < P::REG_M; i++) {
            const size_t mma_trd_m1 = mma_wrp_m + i * P::MMA_M + (lid % 4) * 2;
            const float scale_1 = __half2float(smem_A_scale[mma_trd_m1]);
            const size_t mma_trd_m2 = mma_trd_m1 + 1;
            const float scale_2 = __half2float(smem_A_scale[mma_trd_m2]);
            unroll for (size_t j = 0; j < P::REG_N; j++) {
                cf[i][j][0] += scale_1 * __int2float_rn(c[i][j][0]);
                cf[i][j][1] += scale_2 * __int2float_rn(c[i][j][1]);
                cf[i][j][2] += scale_1 * __int2float_rn(c[i][j][2]);
                cf[i][j][3] += scale_2 * __int2float_rn(c[i][j][3]);
            }
        }
    }
    unroll for (size_t i = 0; i < P::REG_M; i++) {
        unroll for (size_t j = 0; j < P::REG_N; j++) {
            const size_t mma_trd_m = mma_wrp_m + i * P::MMA_M + (lid % 4) * 2;
            const size_t mma_trd_n = mma_wrp_n + j * P::MMA_N + lid / 4;
            const size_t gmem_m = mma_trd_m + gmem_m_base;
            const size_t gmem_n = mma_trd_n + gmem_n_base;
            assert(gmem_m < M && gmem_n < N);
            C[(gmem_m + 0) * N + gmem_n + 0] = __float2half(cf[i][j][0]);
            C[(gmem_m + 1) * N + gmem_n + 0] = __float2half(cf[i][j][1]);
            C[(gmem_m + 0) * N + gmem_n + 8] = __float2half(cf[i][j][2]);
            C[(gmem_m + 1) * N + gmem_n + 8] = __float2half(cf[i][j][3]);
        }
    }
#undef unroll
}
} // namespace

void matmul_host_handwritten(const Int4Storage *A, const Int4Storage *B, uint32_t M, uint32_t N, uint32_t K,
                             int32_t *C) {
    using P = MatMulParams;

    const dim3 dim_block{P::N_THR};
    const dim3 dim_grid(ceil_div(M, P::SHM_M), ceil_div(N, P::SHM_N));
    constexpr size_t shmem_size = MatmulDerivedParams<P>::SHM_SIZE;
    if constexpr (shmem_size > 48 * 1024) {
        ensure(cudaFuncSetAttribute(matmul_kernel<P>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    shmem_size) == cudaSuccess,
               "Failed to set shared memory size");
    }
    matmul_kernel<P><<<dim_grid, dim_block, shmem_size>>>(reinterpret_cast<const uint8_t *>(A),
                                                          reinterpret_cast<const uint8_t *>(B), C, M, N, K);
    ensure(cudaDeviceSynchronize() == cudaSuccess, "Failed to synchronize device");
}

void matmul_hadamard_host(const half *A, const Int4Storage *B, uint32_t M, uint32_t N, uint32_t K, half *C) {
    using P = MatMulHadamardParams;

    const dim3 dim_block{P::N_THR};
    const dim3 dim_grid(ceil_div(M, P::SHM_M), ceil_div(N, P::SHM_N));
    constexpr size_t shmem_size = MatmulHadamardDerivedParams<P>::SHM_SIZE;
    static_assert(shmem_size <= 96 * 1024, "Shared memory size exceeds 96KB");
    if constexpr (shmem_size > 48 * 1024) {
        ensure(cudaFuncSetAttribute(matmul_hadamard_kernel<P>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    shmem_size) == cudaSuccess,
               "Failed to set shared memory size");
    }
    matmul_hadamard_kernel<P>
        <<<dim_grid, dim_block, shmem_size>>>(A, reinterpret_cast<const uint8_t *>(B), C, M, N, K);
    ensure(cudaDeviceSynchronize() == cudaSuccess, "Failed to synchronize device");
}

void matmul_host(const Int4Storage *A, const Int4Storage *B, uint32_t M, uint32_t N, uint32_t K, int32_t *C) {
    using Gemm =
        cutlass::gemm::device::Gemm<cutlass::int4b_t,               // ElementA
                                    cutlass::layout::RowMajor,      // LayoutA
                                    cutlass::int4b_t,               // ElementB
                                    cutlass::layout::ColumnMajor,   // LayoutB
                                    int32_t,                        // ElementOutput
                                    cutlass::layout::RowMajor,      // LayoutOutput
                                    int32_t,                        // ElementAccumulator
                                    cutlass::arch::OpClassTensorOp, // tag indicating Tensor Cores
                                    cutlass::arch::Sm80 // tag indicating target GPU compute architecture //
                                                        // TODO: This is just for compiling on my laptop
                                                        // temporarily. Should be higher when doing
                                                        // benchmarking.
                                    >;

    Gemm gemmOp;

    using GemmCoord = cutlass::gemm::GemmCoord;

    typename Gemm::Arguments arguments{{static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
                                        static_cast<GemmCoord::Index>(K)},
                                       {(cutlass::int4b_t *)A, K},
                                       {(cutlass::int4b_t *)B, K},
                                       {C, N},
                                       {C, N},
                                       {1, 0}};

    auto status = gemmOp(arguments);

    ensure(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
}