#include <cstdint>
#include <cutlass/gemm/device/gemm.h>
#include <gemm.h>

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
__global__ __launch_bounds__(P::N_THR) void matmul_handwritten(const uint8_t *A, const uint8_t *B, int32_t *C,
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

} // namespace

void matmul_host_fused(const Int4Storage *A, const Int4Storage *B, uint32_t M, uint32_t N, uint32_t K,
                       int32_t *C) {
    using P = MatMulParams;

    const dim3 dim_block{P::N_THR};
    const dim3 dim_grid(ceil_div(M, P::SHM_M), ceil_div(N, P::SHM_N));
    constexpr size_t shmem_size = SHM_SIZE<P>;
    if constexpr (shmem_size > 48 * 1024) {
        ensure(cudaFuncSetAttribute(matmul_handwritten<P>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    shmem_size) == cudaSuccess,
               "Failed to set shared memory size");
    }
    matmul_handwritten<P><<<dim_grid, dim_block, shmem_size>>>(
        reinterpret_cast<const uint8_t *>(A), reinterpret_cast<const uint8_t *>(B), C, M, N, K);
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