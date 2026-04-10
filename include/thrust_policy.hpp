#ifndef CBNS_THRUST_POLICY_HPP
#define CBNS_THRUST_POLICY_HPP

#include <thrust/execution_policy.h>

namespace cbns
{
// For Phase 1 we only distinguish CUDA vs non-CUDA.
// If/when you want OMP/TBB, extend this with those execution policies.
#if defined(THRUST_DEVICE_SYSTEM) && (THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA)
#  include <thrust/system/cuda/execution_policy.h>
inline auto thrust_policy()
{
  return thrust::cuda::par;
}
#else
inline auto thrust_policy()
{
  return thrust::cpp::par;
}
#endif
} // namespace cbns

#endif // CBNS_THRUST_POLICY_HPP

