#ifndef H_GEMM_TYPES
#define H_GEMM_TYPES

#include <half.hpp>
using half_float::half;

#if !(defined(HGEMM) || defined(SGEMM) || defined(DGEMM) || defined(HHS) || defined(HSS))
#warning "no GEMM operation specific, using SGEMM by default"
#define HGEMM
#endif

#if defined(HGEMM) || defined(HSS) || defined(HHS)
typedef half typeAB;
const rocblas_datatype ab_type = rocblas_datatype_f16_r;
#elif defined(SGEMM)
typedef float typeAB;
const rocblas_datatype ab_type = rocblas_datatype_f32_r;
#elif defined(DGEMM)
typedef double typeAB;
const rocblas_datatype ab_type = rocblas_datatype_f64_r;
#endif

#if defined(HGEMM) || defined(HHS)
typedef half typeCD;
const rocblas_datatype cd_type = rocblas_datatype_f16_r;
#elif defined(SGEMM) || defined(HSS)
typedef float typeCD;
const rocblas_datatype cd_type = rocblas_datatype_f32_r;
#elif defined(DGEMM)
typedef double typeCD;
const rocblas_datatype cd_type = rocblas_datatype_f64_r;
#endif

#if defined(HGEMM)
typedef half typeAlphaBeta;
const rocblas_datatype compute_type = rocblas_datatype_f16_r;
#elif defined(SGEMM) || defined(HHS) || defined(HSS)
typedef float typeAlphaBeta;
const rocblas_datatype compute_type = rocblas_datatype_f32_r;
#elif defined(DGEMM)
typedef double typeAlphaBeta;
const rocblas_datatype compute_type = rocblas_datatype_f64_r;
#endif

#endif
