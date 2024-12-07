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

struct MatMulHadamardParams {
    constexpr static size_t N_THR = 256;

    constexpr static size_t SHM_M = 64;
    constexpr static size_t SHM_N = 128;
    constexpr static size_t SHM_K = 128;

    constexpr static size_t CPY_K = 32; // Copy at the granularity of 16 bytes

    constexpr static size_t MMA_M = 16;
    constexpr static size_t MMA_N = 8;
    constexpr static size_t MMA_K = 64;

    constexpr static size_t REG_M = 2;
    constexpr static size_t REG_N = 4;
};

template <typename A, typename B> auto ceil_div(A a, B b) { return (a + b - 1) / b; }

template <typename P>
constexpr size_t SHM_K_STRIDE = [] {
    constexpr size_t EPB = 2; // Elements per byte
    static_assert(P::SHM_K % (EPB * sizeof(int32_t)) == 0);
    constexpr size_t SHM_K_INT32S = P::SHM_K / (EPB * sizeof(int32_t));
    static_assert(SHM_K_INT32S >= 4);
    return (((SHM_K_INT32S - 4 + 7) & ~7) + 4) * sizeof(int32_t);
}();

template <typename P> constexpr size_t SHM_A_SIZE = P::SHM_M * SHM_K_STRIDE<P>;
template <typename P> constexpr size_t SHM_B_SIZE = P::SHM_N * SHM_K_STRIDE<P>;
template <typename P> constexpr size_t SHM_SIZE = 2 * (SHM_A_SIZE<P> + SHM_B_SIZE<P>);

template <typename P>
constexpr size_t SHM_SIZE_FUSED = 2 * (SHM_A_SIZE<P> + SHM_B_SIZE<P> + P::SHM_M * sizeof(half));

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

template <typename P>
__global__ __launch_bounds__(P::N_THR) void matmul_kernel(const uint8_t *A, const uint8_t *B, int32_t *C,
                                                          uint32_t M, uint32_t N, uint32_t K) {
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

    extern __shared__ uint8_t smem[];
    uint8_t *__restrict__ smem_A = smem;
    uint8_t *__restrict__ smem_B = smem + SHM_A_SIZE<P> * 2;

    const size_t tid = threadIdx.x;
    const size_t wid = tid / 32;
    const size_t lid = tid % 32;
    const size_t gid = lid / 8;

    uint8_t *__restrict__ smem_A_cur = smem_A, *__restrict__ smem_A_next = smem_A + SHM_A_SIZE<P>;
    uint8_t *__restrict__ smem_B_cur = smem_B, *__restrict__ smem_B_next = smem_B + SHM_B_SIZE<P>;

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
                reinterpret_cast<CopyType *>(smem_A_next + tile_m * SHM_K_STRIDE<P> + tile_k / EPB);
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
                reinterpret_cast<CopyType *>(smem_B_next + tile_n * SHM_K_STRIDE<P> + tile_k / EPB);
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
                    __cvta_generic_to_shared(smem_A_cur + mma_trd_m * SHM_K_STRIDE<P> + mma_trd_k / EPB);
                asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                    : "=r"(a[i][0]), "=r"(a[i][1]), "=r"(a[i][2]), "=r"(a[i][3])
                    : "l"(addr));
            }
            unroll for (size_t j = 0; j < P::REG_N; j++) {
                const size_t mma_trd_n = mma_trd_n_ld_base + j * P::MMA_N;
                const size_t mma_trd_k = k_ + gid * 32;
                const size_t addr =
                    __cvta_generic_to_shared(smem_B_cur + mma_trd_n * SHM_K_STRIDE<P> + mma_trd_k / EPB);
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

/**
 * Fused hadamard-linear kernel with group quantization.
 *
 * Conceptually: fp16 input -> Hadamard -> 16-to-4 bit quantization -> int4 linear -> 32-to-16 quantization
 */
template <typename P>
__global__ __launch_bounds__(P::N_THR) void matmul_hadamard_kernel(const half *A, const uint8_t *B, half *C,
                                                                   uint32_t M, uint32_t N, uint32_t K) {
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

    extern __shared__ uint8_t smem[];
    uint8_t *__restrict__ smem_A = smem;
    uint8_t *__restrict__ smem_B = smem + SHM_A_SIZE<P> * 2;

    uint8_t *__restrict__ smem_A_scale_cur = smem_B + SHM_B_SIZE<P> * 2;
    uint8_t *__restrict__ smem_A_scale_next = smem_A_scale_cur + P::SHM_M * sizeof(half);

    const size_t tid = threadIdx.x;
    const size_t wid = tid / 32;
    const size_t lid = tid % 32;
    const size_t gid = lid / 8;

    uint8_t *__restrict__ smem_A_cur = smem_A, *__restrict__ smem_A_next = smem_A + SHM_A_SIZE<P>;
    uint8_t *__restrict__ smem_B_cur = smem_B, *__restrict__ smem_B_next = smem_B + SHM_B_SIZE<P>;

#define unroll _Pragma("unroll")

    const auto async_copy_to_smem = [&](size_t k) {
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
                reinterpret_cast<CopyType *>(smem_B_next + tile_n * SHM_K_STRIDE<P> + tile_k / EPB);
            async_copy(dst, src);
        }
        // Hadamard to A
        unroll for (size_t i = 0; i < P::SHM_M; i += N_WRP) {
            const size_t tile_m = i + wid;
            const size_t gmem_m = tile_m + blockIdx.x * P::SHM_M;
            half *group_scale = reinterpret_cast<half *>(smem_A_scale_next + tile_m * sizeof(half));
            hadamard_transform_group_quantize<P::SHM_K, 32>(
                A + gmem_m * K + k, smem_A_next + tile_m * SHM_K_STRIDE<P>, group_scale);
        }
    };

    float cf[P::REG_M][P::REG_N][4] = {0};
    int32_t c[P::REG_M][P::REG_N][4] = {0};

    const size_t mma_wrp_m = wid / WRP_N * P::REG_M * P::MMA_M;
    const size_t mma_wrp_n = wid % WRP_N * P::REG_N * P::MMA_N;
    const size_t mma_trd_m_ld_base = mma_wrp_m + lid % 8 + (gid % 2) * 8;
    const size_t mma_trd_n_ld_base = mma_wrp_n + lid % 8;

    async_copy_to_smem(0);

    for (size_t k = 0; k < K; k += P::SHM_K) {
        swap(smem_A_cur, smem_A_next);
        swap(smem_B_cur, smem_B_next);
        swap(smem_A_scale_cur, smem_A_scale_next);
        async_copy_waitall();

        /*
        if (threadIdx.x == 0) {
            constexpr size_t SHOW = 10;
            const auto print_row = [&](size_t i) {
                printf("        [");
                const auto to_int4 = [](int8_t a) { return a > 7 ? a - 16 : a; };
                for (size_t j = 0; j < SHOW; j += 2) {
                    uint8_t packed = smem_A_cur[i * SHM_K_STRIDE<P> + j / EPB];
                    printf("%2d, %2d, ", to_int4(packed & 0xf), to_int4(packed >> 4));
                }
                printf(" ..., ");
                for (size_t j = P::SHM_K - SHOW; j < P::SHM_K; j += 2) {
                    uint8_t packed = smem_A_cur[i * SHM_K_STRIDE<P> + j / EPB];
                    printf("%2d, %2d", to_int4(packed & 0xf), to_int4(packed >> 4));
                    if (j + 2 < P::SHM_K)
                        printf(", ");
                }
                printf("],\n");
            };
            for (size_t i = 0; i < SHOW; i++)
                print_row(i);
            printf("        ...,\n");
            for (size_t i = P::SHM_M - SHOW; i < P::SHM_M; i++)
                print_row(i);
        }
        */

        __syncthreads();
        if (k + P::SHM_K < K) {
            async_copy_to_smem(k + P::SHM_K);
        }

        // Zero out the accumulation buffer
        unroll for (size_t i = 0; i < P::REG_M; i++) {
            unroll for (size_t j = 0; j < P::REG_N; j++) {
                unroll for (size_t k_ = 0; k_ < 4; k_++) { c[i][j][k_] = 0; }
            }
        }
        unroll for (size_t k_ = 0; k_ < P::SHM_K; k_ += P::MMA_K) {
            int32_t a[P::REG_M][4], b[P::REG_N][2];
            unroll for (size_t i = 0; i < P::REG_M; i++) {
                const size_t mma_trd_m = mma_trd_m_ld_base + i * P::MMA_M;
                const size_t mma_trd_k = k_ + (gid / 2) * 32;
                const size_t addr =
                    __cvta_generic_to_shared(smem_A_cur + mma_trd_m * SHM_K_STRIDE<P> + mma_trd_k / EPB);
                asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                    : "=r"(a[i][0]), "=r"(a[i][1]), "=r"(a[i][2]), "=r"(a[i][3])
                    : "l"(addr));
            }
            unroll for (size_t j = 0; j < P::REG_N; j++) {
                const size_t mma_trd_n = mma_trd_n_ld_base + j * P::MMA_N;
                const size_t mma_trd_k = k_ + gid * 32;
                const size_t addr =
                    __cvta_generic_to_shared(smem_B_cur + mma_trd_n * SHM_K_STRIDE<P> + mma_trd_k / EPB);
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
        unroll for (size_t i = 0; i < P::REG_M; i++) {
            const size_t mma_trd_m1 = mma_wrp_m + i * P::MMA_M + lid / 4;
            const float scale_1 = __half2float(reinterpret_cast<half *>(smem_A_scale_cur)[mma_trd_m1]);
            const size_t mma_trd_m2 = mma_trd_m1 + 8;
            const float scale_2 = __half2float(reinterpret_cast<half *>(smem_A_scale_cur)[mma_trd_m2]);
            unroll for (size_t j = 0; j < P::REG_N; j++) {
                cf[i][j][0] += scale_1 * __int2float_rn(c[i][j][0]);
                cf[i][j][1] += scale_1 * __int2float_rn(c[i][j][1]);
                cf[i][j][2] += scale_2 * __int2float_rn(c[i][j][2]);
                cf[i][j][3] += scale_2 * __int2float_rn(c[i][j][3]);
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
            C[(gmem_m + 0) * N + gmem_n + 0] = __float2half(cf[i][j][0]);
            C[(gmem_m + 0) * N + gmem_n + 1] = __float2half(cf[i][j][1]);
            C[(gmem_m + 8) * N + gmem_n + 0] = __float2half(cf[i][j][2]);
            C[(gmem_m + 8) * N + gmem_n + 1] = __float2half(cf[i][j][3]);
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
    constexpr size_t shmem_size = SHM_SIZE<P>;
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
    constexpr size_t shmem_size = SHM_SIZE_FUSED<P>;
    if constexpr (shmem_size > 48 * 1024) {
        ensure(cudaFuncSetAttribute(matmul_kernel<P>, cudaFuncAttributeMaxDynamicSharedMemorySize,
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