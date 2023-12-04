#include "cuda/std/utility"

struct Add
{
  template <typename T, typename U>
  __host__ __device__
    typename ::cuda::std::common_type<T, U>::type
    operator()(T &&t, U &&u) const
  {
    return t + u;
  }
};

template __host__ __device__
::cuda::std::common_type<int, int>::type
Add::operator()(int &&t, int &&u) const;
