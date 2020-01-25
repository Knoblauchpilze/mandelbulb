#ifndef    CUDA_JOB_HXX
# define   CUDA_JOB_HXX

# include "CudaJob.hh"

namespace utils {

  inline
  CudaJob::CudaJob(const std::string& name,
                   const Priority& priority):
    CoreObject(name),

    m_priority(priority)
  {
    setService("job");
  }

  inline
  Priority
  CudaJob::getPriority() const noexcept {
    return m_priority;
  }

}

#endif    /* CUDA_JOB_HXX */
