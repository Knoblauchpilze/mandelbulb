#ifndef    CUDA_EXECUTOR_HXX
# define   CUDA_EXECUTOR_HXX

# include "CudaExecutor.hh"

namespace utils {

  inline
  CudaExecutor::~CudaExecutor() {
    // Terminate the threads.
    terminateThreads();

    // Release gpu resources.
    destroyCudaSchedulingData();
  }

  inline
  bool
  CudaExecutor::hasJobs() const noexcept {
    return !m_hPrioJobs.empty() || !m_nPrioJobs.empty() || !m_lPrioJobs.empty();
  }

}

#endif    /* CUDA_EXECUTOR_HXX */
