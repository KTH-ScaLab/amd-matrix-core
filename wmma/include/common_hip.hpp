#include <hip/hip_runtime.h>

#if defined(__HIP_PLATFORM_AMD__)
    #include <rocwmma/rocwmma.hpp>
#elif defined(__HIP_PLATFORM_NVIDIA__)
    #include <mma.h>
#else
    #error "unknown platform"
#endif

#ifndef COMMON_HIP_H
#define COMMON_HIP_H

#define CHECK_HIP_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           int const line)
{
    if (err != hipSuccess)
    {
        std::cerr << "HIP Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << hipGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

constexpr int log2(int n) {
    int res = 0;
    while(n >>= 1) res++;
    return res;
};

#endif
