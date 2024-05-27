# Benchmarks and Tools for Characterizing AMD Matrix Cores

This repository contains benchmarking tools targeting performance and power evaluation
of 2nd generation AMD Matrix Cores, available on *e.g.* AMD Instinct MI250X accelerator.

Prerequisites:
- HIP (rocm-5.3.3)
- ROCm 5.3.3

## Cite Publications
Gabin Schieffer, Daniel Medeiros, Jennifer Faj, Aniruddha Marathe, Ivy Peng. 2024. [*On the Rise of AMD Matrix Cores: Performance, Power Efficiency, and Programmability*](https://www.osti.gov/servlets/purl/2345982). IEEE IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS) 2024.

## Power Sampling
The `power-sampler` folder provides a simple power sampling tool to measure the power consumption of an AMD GPU with a user-controlled sampling period. The underlying interface used to perform this measurement is `rsmi_dev_power_ave_get`, which is part of AMD's ROCm System Management Interface (ROCm SMI).

### Building
`cd power-sampler && make power_sampler`

### Usage
`./power_sampler sample_period_ms`
- `sample_period_ms`: sampling period, in miliseconds. 100 ms generally provides acceptable results.

The environment variable `HIP_VISIBLE_DEVICES` can be used to set the GCD which measurements are taken from. The tool performs measurements on the first visible GPU.

*Note*: on a single MI250X GPU, only GCD0 provides readings for power consumption. This value is the power consumption of both GCDs on the GPU. As a result, on a node with 4 GPUs (8 GCDs), only GCD0, GCD2, GCD4, and GCD6 will have non-zero values of power consumption, when using this tool.

### Output
The tool outputs data in comma-separated format, one measurement per line: `timestamp,power`, where `timestamp` is a UNIX timestamp in seconds, and `power` is the GCD's measured power average consumption in Watts.

*Note*: as the sampling period depends on measurement latency, it might vary during execution. When analyzing results, users should not assume a constant sampling period, and should instead use the provided measurement timestamp as the time reference.

## Microbenchmark (WMMA)

The `wmma` folder contains code for a micro-benchmark approach to evaluate performance of Matrix Cores on MI250X.

### Building

rocWMMA header files must be accessible to the compiler, either system-wide, or in the local `wmma/include` folder. This can be done by copying the folder `library/include/rocwmma` from the official [rocWMMA repository](https://github.com/ROCm/rocWMMA/releases) to `wmma/include`. We recommend using the rocWMMA version matching the ROCm installation, see https://github.com/ROCm/rocWMMA/releases.

We provide a Makefile for building for both AMD and Nvidia backends. If nvcc is present on the system, Nvidia backend is assumed. The Makefile should be adapted to fit user and system requirements.

Compile-time parameters, such as datatypes, and matrix sizes, are indicated within `CHANGEME` marks in the file `src/wmma_gemm.cpp`.

Compile with `cd wmma && make bin/wmma_gemm`.

*Note*: the make target `bin/wmma_gemm.s` allows obtaining assembly code for the the micro-benchmark.

### Usage
`bin/./wmma_gemm n_blocks warps_per_block`

- `n_blocks`: number of blocks to use as kernel launch configuration.
- `warps_per_block`: number of warps per block. On the AMD platform, this refers to a _wavefront_, consisting of 64 threads. On MI250X, we recommend a value of 4 to accomodate for the four Matrix Cores per Compute Unit.

### Output
The values of each parameter in the micro-benchmark are printed. Statistics on WMMA iterations are printed (average, minimum, maximum, standard deviation), along with kernel execution time, and measured throughput.

## High-level GEMM (BLAS)

### Building
We provide a Makefile for building for both AMD and Nvidia backends. If nvcc is present on the system, Nvidia backend is assumed. The Makefile should be adapted to fit user and system requirements.

Compile with `cd blas && make bin/gemm_blas GEMM_OP=<GEMM_OP>`.

`<GEMM_OP>` is the GEMM operation which is used to create the benchmark binary, among `DGEMM`, `HGEMM`, `SGEMM`, `HHS`, `HSS`. Those operations are detailed in `include/gemm_types.hpp`. We further refer to rocBLAS official documentation for details, in particular on the dataypes used for each operand (https://rocblas.readthedocs.io/en/master/Programmers_Guide.html#id16).

Additionnaly, the `bin/gemm_blas.s` target can be use to generate assembly. This enables inspecting compiler-generated code, and identify which Matrix Core instructions are being used.

### Usage
``bin/./gemm_blas N``
- `N`: each matrix operand is of shape `N`*`N`.

Environment variables:
- `NO_ACCUM`: when set, disables accumulation. This option increases memory footprint.
- `B_IS_A`: when set, `A == B`. This options reduces memory footprint, allowing to increase matrix size;
- `N_EXP`: number of repetitions (default: 100).

*Note*: when `N < 512`, the result of rocBLAS GEMM is checked against a CPU-computed reference value.

### Output

The code produces a comma-separated list of time measurements, in miliseconds, one per GEMM repetition.

## Scripts
The folder `scripts` contains various scripts to execute, profile tools and perform result analysis.

### run_blas.sh
Execute profiling on the rocBLAS benchmark. This script uses rocprof to obtain the number of floating-point operation performed by regular SIMD units, and Matrix Core units. The results are placed into the file `$PROF_OUTFILE` (by default `tmp.csv`).

### measure_power.sh
Performs power measurements using our power sampling tools, when executing two instances of the rocWMMA micro-benchmark in parallel on two GCDs of the system, for various kernel launch configurations. The output is a comma-separated format, in `timestamp,power` format, as produced by our power sampling tool. 

In this script `$GPU_ID` sets the ID of the GCD on which power measurements are performed. The micro-benchmark is executed in parallel on two GCDs: `$GPU_ID` and `$GPU_ID+1`.`$GPU_ID` must refer to a primary GCD on the system. Thus, it must always be an even number. For example, on a system with four MI250X GPUs, this can be 0, 2, 4, or 6.

## Open-source components

**half - IEEE 754-based half-precision floating point library.** <br>
Copyright (c) 2012-2017 Christian Rau \<rauy@users.sourceforge.net\> <br>
https://half.sourceforge.net/

**rocBLAS-Examples**<br>
Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved. <br>
https://github.com/ROCm/rocBLAS-Examples

**rocWMMA**<br>
Copyright 2016-2022 Advanced Micro Devices, Inc. <br>
https://github.com/ROCm/rocWMMA

## Contact
Gabin Schieffer <gabins@kth.se>
Ivy Peng <ipeng@acm.org>
