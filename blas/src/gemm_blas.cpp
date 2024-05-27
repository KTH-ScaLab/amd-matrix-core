#include <common_hip.hpp>
#include <gemm_types.hpp>

#include <vector>
#include <string>
#include <chrono>

using namespace std::chrono;

template <typename T>
void matIdentity(T* A, int M, int N, size_t lda) {
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            A[i + j * lda] = T(i == j);
}

template <typename T1, typename T2>
void matMatMult(typeAlphaBeta  alpha,
                typeAlphaBeta  beta,
                int M,
                int N,
                int K,
                T1*  A,
                int As1,
                int As2,
                T1*  B,
                int Bs1,
                int Bs2,
                T2*  C,
                int Cs1,
                int Cs2)
{
    for(int i1 = 0; i1 < M; i1++) {
        for(int i2 = 0; i2 < N; i2++) {
            T1 t = T1(0.0);
            for(int i3 = 0; i3 < K; i3++)
                t += A[i1 * As1 + i3 * As2] * B[i3 * Bs1 + i2 * Bs2];
            C[i1 * Cs1 + i2 * Cs2] = beta * C[i1 * Cs1 + i2 * Cs2] + alpha * t;
        }
    }
}

template <typename T>
double maxRelativeError(const T* A, const T* reference, size_t n) {
    double maxRelativeError = double(std::numeric_limits<T>::min());

    for(size_t i = 0; i < n; ++i) {
        double gold          = double(reference[i]);
        double relativeError = gold != 0 ? (gold - double(A[i])) / (gold) : double(A[i]);
        relativeError        = relativeError > 0 ? relativeError : -relativeError;
        maxRelativeError = relativeError < maxRelativeError ? maxRelativeError : relativeError;
    }
    return maxRelativeError;
}

template <typename T>
void displayMat(const T* mat, size_t M, size_t N) {
    for(size_t i = 0; i < M; i++) {
        for(size_t j = 0; j < N; j++)
            std::cout << mat[j + i*M] << " ";

        std::cout << std::endl;
    }
}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " N" << std::endl;
        exit(1);
    }
    long long n_argv = std::stoi(argv[1]);

    /* NO_ACCUM:
     * By default, results of consecutive GEMM operations are accumulated in C:
     *      C <- alpha * AB + beta * C
     * setting NO_ACCUM environment variable will disable this behavior:
     *      D <- alpha * AB + beta * C
     */
    bool no_accum = std::getenv("NO_ACCUM");
    bool b_is_a = std::getenv("B_IS_A");

    int N_EXP = 100;
    if(const char *c = std::getenv("N_EXP"))
        N_EXP = std::stoi(c);

    /* parse arguments */
    long long M = n_argv; // std::stoi(argv[1]);
    long long N = n_argv; // std::stoi(argv[2]);
    long long K = n_argv; // std::stoi(argv[3]);
    typeAlphaBeta hAlpha = static_cast<typeAlphaBeta>(0.1); // std::stof(argv[4]);
    typeAlphaBeta hBeta  = static_cast<typeAlphaBeta>(0.1); // std::stof(argv[5]);
    
    /* init rocblas API */
    rocblas_status rstatus = rocblas_status_success;
    rocblas_handle handle;
    CHECK_ROCBLAS_STATUS( rocblas_create_handle(&handle) );
    
    /* get rocBLAS stream */
    hipStream_t stream;
    CHECK_ROCBLAS_STATUS( rocblas_get_stream(handle, &stream) );

    /* init rocBLAS parameters */
    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    long long lda, ldb, ldc, sizeA, sizeB, sizeC;
    int          strideA1, strideA2, strideB1, strideB2;

    if(transA == rocblas_operation_none) {
        lda      = M;
        sizeA    = K * lda;
        strideA1 = 1;
        strideA2 = lda;
    }
    else {
        lda      = K;
        sizeA    = M * lda;
        strideA1 = lda;
        strideA2 = 1;
    }
    if(transB == rocblas_operation_none) {
        ldb      = K;
        sizeB    = N * ldb;
        strideB1 = 1;
        strideB2 = ldb;
    }
    else {
        ldb      = N;
        sizeB    = K * ldb;
        strideB1 = ldb;
        strideB2 = 1;
    }
    ldc   = M;
    sizeC = N * ldc;

    typeAB *hA, *hB;
    typeCD *hC, *hGold;
    hA    = (typeAB*) malloc(sizeof(typeAB) * sizeA);
    hC    = (typeCD*) malloc(sizeof(typeCD) * sizeC);
    hGold = (typeCD*) malloc(sizeof(typeCD) * sizeC);
    if(b_is_a)
        hB = hA;
    else
        hB = (typeAB*) malloc(sizeof(typeAB) * sizeB);

    if(hA == NULL || hB == NULL || hC == NULL || hGold == NULL) {
        std::cerr << "malloc error" << std::endl;
        exit(2);
    }

    /* init A, B, C */ 
    for(size_t i = 0; i < sizeA; i++)
        hA[i] = static_cast<typeAB>(1.0);
    for(size_t i = 0; i < sizeC; i++)
        hC[i] = hGold[i] = static_cast<typeCD>(0.0);
    matIdentity(hB, K, N, ldb);
    
    /* init HIP runtime */
    int device;
    CHECK_HIP_ERROR( hipGetDevice(&device) );

    hipDeviceProp_t prop;
    CHECK_HIP_ERROR( hipGetDeviceProperties(&prop, device) );
    
    std::cout
        << "-- GPU #" << device << ": " << prop.name << std::endl
        << "M=" << M << ",N=" << N << ",K=" << K << std::endl
        << "NO_ACCUM=" << no_accum << std::endl
        << "B_IS_A=" << b_is_a << std::endl
        << "N_EXP=" << N_EXP << std::endl;

    /* enable passing alpha parameter from pointer to host memory */
    CHECK_ROCBLAS_STATUS( rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host) );
    
    /* allocate on device, and copy from host */
    typeAB *dA, *dB;
    typeCD *dC, *dD;
    CHECK_HIP_ERROR( hipMalloc(&dA, sizeA * sizeof(typeAB)) );
    CHECK_HIP_ERROR( hipMalloc(&dC, sizeC * sizeof(typeCD)) );
    
    if(b_is_a)  dB = dA;
    else        CHECK_HIP_ERROR( hipMalloc(&dB, sizeB * sizeof(typeAB)) );
    
    if(no_accum) CHECK_HIP_ERROR( hipMalloc(&dD, sizeC * sizeof(typeCD)) );
    else         dD = dC;

    CHECK_HIP_ERROR( hipMemcpy(dA, hA, sizeof(typeAB) * sizeA, hipMemcpyHostToDevice) );
    CHECK_HIP_ERROR( hipMemcpy(dB, hB, sizeof(typeAB) * sizeB, hipMemcpyHostToDevice) );
    CHECK_HIP_ERROR( hipMemcpy(dC, hC, sizeof(typeCD) * sizeC, hipMemcpyHostToDevice) );

    /* avoid cold start, and allows performing test */
    rstatus = rocblas_gemm_ex(
            handle, transA, transB, M, N, K, 
            &hAlpha,
            dA, ab_type, lda,
            dB, ab_type, ldb,
            &hBeta,
            dC, cd_type, ldc,
            dD, cd_type, ldc,
            compute_type, rocblas_gemm_algo_standard, NULL, rocblas_gemm_flags_none);
    hipStreamSynchronize(stream);
    
    CHECK_ROCBLAS_STATUS(rstatus);
    
    /* Check relative error */
    if(N <= 512) {
        CHECK_HIP_ERROR( hipMemcpy(hC, dD, sizeof(typeCD) * sizeC, hipMemcpyDeviceToHost) );

        matMatMult(hAlpha, hBeta, M, N, K,
                   hA, strideA1, strideA2,
                   hB, strideB1, strideB2,
                   hGold, 1, ldc);

        typeCD err = (typeCD) maxRelativeError(hC, hGold, M*N);
        typeCD eps = std::numeric_limits<typeCD>::epsilon();
        typeCD tol = (typeCD) 10;

        std::cerr
            << ((err > eps*tol) ? "FAIL" : "PASS")
            << ": max relative err. = " << err << std::endl;
    }
    else
        std::cerr << "SKIPPED: test skipped" << std::endl;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
for(int t = 0; t < N_EXP; t++) {
    auto host_start = high_resolution_clock::now();

    // hipEventRecord(start, stream);
    rstatus = rocblas_gemm_ex(
            handle, transA, transB, M, N, K, 
            &hAlpha,
            dA, ab_type, lda,
            dB, ab_type, ldb,
            &hBeta,
            dC, cd_type, ldc,
            dD, cd_type, ldc, 
            /* compute_type */ compute_type, 
            /* algo */ rocblas_gemm_algo_standard,
            /* solution_index */ NULL,
            /* flags */ NULL);
    // hipEventRecord(stop, stream);

    hipStreamSynchronize(stream);

    auto host_stop = high_resolution_clock::now();
    float elapsed_ms_host = 
        (float) duration_cast<microseconds>(host_stop - host_start).count() / 1000;

    std::cout << elapsed_ms_host << ",";
    
    // float elapsed_ms_device;
    // hipEventElapsedTime(&elapsed_ms_device, start, stop);
    CHECK_ROCBLAS_STATUS( rstatus );
}
    std::cout << std::endl;


    /* destroy */
    CHECK_ROCBLAS_STATUS( rocblas_destroy_handle(handle) );

    free(hA);
    if(!b_is_a) free(hB);
    free(hC);
    free(hGold);

    hipFree(dA);
    if(!b_is_a) hipFree(dB);
    hipFree(dC);
    if(dC != dD) hipFree(dD);

    hipEventDestroy(start);
    hipEventDestroy(stop);

    std::cout << "done" << std::endl;
    return 0;
}
