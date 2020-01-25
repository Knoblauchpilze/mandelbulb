#ifndef    CUDA_JOB_HH
# define   CUDA_JOB_HH

# include <memory>
# include <core_utils/CoreObject.hh>
# include <core_utils/JobPriority.hh>

namespace utils {

  class CudaJob: public CoreObject {
    public:

      /**
       * @brief - Retrieve the priority associated to this job. We assume that the priority
       *          cannot be modified once the job has been created hence the fact that we
       *          are allowed to access it without locking anything.
       * @return - the priority associated to this job.
       */
      Priority
      getPriority() const noexcept;

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
       * @brief - Interface method describing the dimensions of the result buffer to allocate
       *          for this job. It is an indication of the amount of data that should be used
       *          by the scheduler to execute the job in the cuda kernels. Combined with the
       *          `getOutputDataSize` one can determine precisely the amount of memory that
       *          will have to be reserved to correctly execute the job.
       * @return - a size describing the dimensions of the output buffer needed by this job.
       */
      virtual utils::Sizei
      getOutputSize() = 0;

      /**
       * @brief - Interface method allowing to retrieve the amount of memory needed to store
       *          a single element of the result buffer expected by this job. The size uses
       *          bytes and should be sufficient to store the data produced by the cuda job
       *          computing the data for this tile.
       *          Used and checked internally by the cuda executor to make sure that it is
       *          consistent with its reserved space and to actually pass the data to the
       *          jobs.
       */
      virtual unsigned
      getOutputDataSize() = 0;

      /**
       * @brief - Interface method allowing to retrieve a pointer to the input parameter to
       *          pass to the kernel executing this job. This should point to a contiguous
       *          block of memory of `getInputDataSize` bytes.
       *          Used by the scheduler to populate the cuda kernel's argument.
       * @return - a pointer to the parameter's data for this job.
       */
      virtual void*
      getInputData() = 0;

      /**
       * @brief - Interface method allowing to retrieve a pointer to the output buffer where
       *          results of the cuda kernel's execution should be saved. This should point
       *          to a contiguous memory area of `getOutputDataSize() * getOutputSize()`
       *          bytes where the cuda scheduler can copy back the result.
       * @return - a pointer to the output result buffer for this job.
       */
      virtual void*
      getOutputData() = 0;

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
