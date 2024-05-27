#include <iostream>

#include "common_hip.hpp"
#include <cmath>

// some macros to print typenames
#define xstr(s) str(s)
#define str(s) #s

/* --- CHANGEME: PARAMETERS --- */
#define M 16
#define N 16
#define K 16

#define ITERS (1024*4096*8)

#define OP_T  half
#define ACC_T float
/* ------- END CHANGEME ------- */

template<typename TOp, typename TAcc, int WMMA_M, int WMMA_N, int WMMA_K>
__device__ void wmma_gemm(
        TOp const* A, TOp const* B, TAcc* C, 
        TOp const alpha, TAcc const beta,
        uint64_t* ts, int iters) {
   
#if defined(__HIP_PLATFORM_AMD__)
    using namespace rocwmma;
#elif defined(__HIP_PLATFORM_NVIDIA__)
    using namespace nvcuda::wmma;
#else
    #error "unknown HIP platform, no known WMMA implementation"
#endif

    /* D = alpha.AB + beta.C */
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, TOp, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, TOp, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, TAcc> c_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, TAcc> d_frag;

    const size_t globalWarpIdx = 
        (blockIdx.x*blockDim.x + threadIdx.x) / warpSize;
 
    TOp const* my_A = A; // + globalWarpIdx*M*K;
    TOp const* my_B = B; // + globalWarpIdx*K*N;
    load_matrix_sync(a_frag, my_A, WMMA_K);
    load_matrix_sync(b_frag, my_B, WMMA_K);
    fill_fragment(c_frag, static_cast<TAcc>(0));
    fill_fragment(d_frag, static_cast<TAcc>(0));

    uint64_t start_time, end_time;
    start_time = clock64();
#pragma unroll
    for(int i = 0; i < ITERS; i++) {
        // we accumulate in D, otherwise this is optimized away.
        mma_sync(d_frag, a_frag, b_frag, d_frag);
    }
    end_time = clock64(); 

    // prevent optimisation, and allow reading the result
    if(blockIdx.x == 0)
        store_matrix_sync(C, d_frag, WMMA_N, mem_row_major);

    int lane = threadIdx.x & (warpSize - 1);
    if(lane == 0)
        ts[globalWarpIdx] = end_time - start_time;
}

template<typename TOp, typename TAcc>
__global__ void gemm_kernel(uint64_t *ts, TOp const* A, TOp const* B, TAcc* C, int iters) {
    wmma_gemm<TOp, TAcc, M, N, K>(
            A, B, C, static_cast<TOp>(1), static_cast<TAcc>(1), ts, iters);
}

template<typename T>
__global__ void fill_values(T* mat, int const n) {
    for(int i = threadIdx.x; i < n; i += blockDim.x)
        mat[i] = ( ((i / M) ^ i) & 1 ) ? 
            static_cast<T>(0) :
            static_cast<T>(1.0 / (4 * (i + 1)));
}

template <typename T>
void print_mat(T const* mat, int const m, int const n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++)
           printf("%f ", (float) mat[n*i+j]);
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    if(argc != 3) {
        std::cerr << "usage: " << argv[0] << " n_blocks warps_per_block" << std::endl;
        exit(1);
    }
    const int blocks          = std::atoi(argv[1]);
    const int warps_per_block = std::stoi(argv[2]);

    int device;
    CHECK_HIP_ERROR( hipGetDevice(&device) );
    
    hipDeviceProp_t prop;
    CHECK_HIP_ERROR( hipGetDeviceProperties(&prop, device) );

    const int threads = prop.warpSize * warps_per_block;
    const int iters = ITERS; // DO NOT MODIFY
    const int N_MAT = warps_per_block * blocks;

    std::cout
        << "-- GPU #" << device << ": " << prop.name << std::endl
        << "clockRate=" << prop.clockRate / 1000 << " MHz" << std::endl
        << "iters=" << iters << std::endl
        << "m=" << M << ",n=" << N << ",k=" << K << std::endl
        << "blocks=" << blocks << std::endl
        << "threads=" << threads << std::endl
        << "typeof(AB)=" xstr(OP_T) << std::endl
        << "typeof(CD)=" xstr(ACC_T) << std::endl;
    
    uint64_t ts[N_MAT], *d_ts;
    CHECK_HIP_ERROR( hipMalloc(&d_ts, N_MAT*sizeof(uint64_t)) );

    OP_T *d_A, *d_B;
    CHECK_HIP_ERROR( hipMalloc(&d_A, M*K*sizeof(OP_T)) );
    CHECK_HIP_ERROR( hipMalloc(&d_B, K*N*sizeof(OP_T)) );
    
    ACC_T *d_C;
    CHECK_HIP_ERROR( hipMalloc(&d_C, M*N*sizeof(ACC_T)) );

    fill_values<OP_T><<<1, threads, 0, 0>>>(d_A, M*K);
    fill_values<OP_T><<<1, threads, 0, 0>>>(d_B, K*N);
    CHECK_HIP_ERROR( hipDeviceSynchronize() );
   
    OP_T A[M*K];
    CHECK_HIP_ERROR( hipMemcpy(A, d_A, M*K*sizeof(OP_T), hipMemcpyDeviceToHost) );
    // print_mat<OP_T>(A, M, K);
    
    hipEvent_t start, stop;
    CHECK_HIP_ERROR( hipEventCreate(&start) );
    CHECK_HIP_ERROR( hipEventCreate(&stop) );
    
    CHECK_HIP_ERROR( hipEventRecord(start, NULL) );
    gemm_kernel<OP_T, ACC_T><<<blocks, threads, 0, 0>>>(d_ts, d_A, d_B, d_C, iters);
    CHECK_HIP_ERROR( hipEventRecord(stop, NULL) );

    CHECK_HIP_ERROR( hipDeviceSynchronize() );
    CHECK_HIP_ERROR( hipMemcpy(ts, d_ts, N_MAT*sizeof(uint64_t), hipMemcpyDeviceToHost) );

    ACC_T C[M*N];
    CHECK_HIP_ERROR( hipMemcpy(C, d_C, M*N*sizeof(ACC_T), hipMemcpyDeviceToHost) );
    // print_mat<ACC_T>(C, M, N);

    float elapsed_ms;
    CHECK_HIP_ERROR( hipEventElapsedTime(&elapsed_ms, start, stop) );
    CHECK_HIP_ERROR( hipEventDestroy(start) );
    CHECK_HIP_ERROR( hipEventDestroy(stop) );

    CHECK_HIP_ERROR( hipFree(d_ts) );
    CHECK_HIP_ERROR( hipFree(d_A) );
    CHECK_HIP_ERROR( hipFree(d_B) );
    CHECK_HIP_ERROR( hipFree(d_C) );

    double avg, stdev, s, tflops;
    uint64_t t_min, t_max;

    // compute average, min, max
    s = t_min = t_max = ts[0];
    for(int i = 1; i < N_MAT; i++){
        s  += ts[i];

        if(ts[i] < t_min) t_min = ts[i];
        else if(ts[i] > t_max) t_max = ts[i];
    }
    avg = (double) s / iters / N_MAT;

    // compute stdev
    s = 0.0;
    for(int i = 0; i < N_MAT; i++) {
        s += pow((double) ts[i] / iters - avg, 2); 
    }
    stdev = sqrt(s / N_MAT);
    
    tflops = (double) 2.0 * N_MAT * M * N * K * iters / elapsed_ms / 1e9;

    std::cout << "--- Timing ---" << std::endl;
    std::cout << "per WMMA:"
        << " avg=" << avg 
        << " min=" << (double) t_min / iters
        << " max=" << (double) t_max / iters
        << " stdev=" << stdev << std::endl;
    std::cout << "kernel: " << elapsed_ms << " ms" 
        << " (" << tflops << " TFLOPS)" << std::endl;

}
