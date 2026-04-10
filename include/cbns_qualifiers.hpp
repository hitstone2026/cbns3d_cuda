#ifndef CBNS_QUALIFIERS_HPP
#define CBNS_QUALIFIERS_HPP

// Qualifiers for functors / inline functions that must compile on both
// host-only and CUDA builds. Not intended to "port" __global__ kernels.

#if defined(__CUDACC__)
#  define CBNS_H  __host__
#  define CBNS_D  __device__
#  define CBNS_HD __host__ __device__
#else
#  define CBNS_H
#  define CBNS_D
#  define CBNS_HD
#endif

#endif // CBNS_QUALIFIERS_HPP

