#ifndef    CUDA_JOB_HH
# define   CUDA_JOB_HH

# include <memory>
# include <core_utils/CoreObject.hh>
# include <core_utils/JobPriority.hh>

namespace utils {

  class CudaJob: public CoreObject {
    public:

      /**
       * @brief - Interface method allowing to retrieve the amount of memory needed to store
       *          the input data for this job. This size is expressed in bytes and should be
       *          sufficient to store all the data that will be passed as input arguments of
       *          the job.
       *          Used and checked internally by the cuda executor to make sure that it is
       *          consistent with its reserved space and to actually pass the data to the
       *          jobs.
       */
      virtual unsigned
      getInputDataSize() = 0;

      /**
       * @brief - Retrieve the priority associated to this job. We assume that the priority
       *          cannot be modified once the job has been created hence the fact that we
       *          are allowed to access it without locking anything.
       * @return - the priority associated to this job.
       */
      Priority
      getPriority() const noexcept;

    protected:

      /**
       * @brief - Creates a new job with the specified priority. The default priority
       *          is set to normal. This constructor is only accessible to inheriting
       *          classes so as to provide some sort of security for who can create a
       *          new job.
       * @param name - the name of this job. Used to provide decent logging.
       * @param priority - the priority of the job to create.
       */
      CudaJob(const std::string& name,
              const Priority& priority = Priority::Normal);

    private:

      /**
       * @brief - The priority associated to this job.
       */
      Priority m_priority;
  };

  using CudaJobShPtr = std::shared_ptr<CudaJob>;
}

# include "CudaJob.hxx"

#endif    /* CUDA_JOB_HH */
