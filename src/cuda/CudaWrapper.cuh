#ifndef    CUDA_WRAPPER_CUH
# define   CUDA_WRAPPER_CUH

# include <mutex>
# include <vector>
# include <functional>
# include <cuda_runtime.h>
# include <core_utils/CoreObject.hh>

namespace utils {

  namespace cuda {

    /**
     * @brief - Convenience define to refer to a cuda stream. This provides some sort
     *          of masking and abstraction on the actual type of a cuda stream because
     *          no one apart from this wrapper really needs to know the underlying
     *          type of such an object.
     */
    using stream_t = void*;

  }

  class CudaWrapper: public CoreObject {
    public:

      /**
       * @brief - Create a new cuda wrapper allowing to wrap cuda API calls and provide
       *          some error checking.
       */
      CudaWrapper();

      virtual ~CudaWrapper();

      /**
       * @brief - Attempts to perform the launch of the input function on the specified
       *          stream. The function checks whether the launch is successful before
       *          returning and indicates it to the caller.
       * @param stream - the stream onto which the function should be performed.
       * @param func - the function to launch on the stream.
       * @return - `true` if the function was successfully launched on the stream and
       *           `false` if something wrong occurred.
       */
      bool
      launch(cuda::stream_t stream,
             std::function<cudaError_t(void)> func);

      /**
       * @brief - Perform the creation of a new cuda stream to be used to schedule
       *          operations on the gpu. The return value is guaranteed to be valid
       *          if the function returns.
       * @param success - if provided will be filled with `true` if the stream could
       *                  successfully be created and `false` otherwise.
       * @return - the created stream.
       */
      cuda::stream_t
      create(bool* success);

      /**
       * @brief - Used to check whether this stream is still up and running. When the
       *          stream is used to schedule some operations, the cuda API can fail
       *          to satisfy the request of the user or the kernel itself can fail to
       *          complete (because of trying to access invalid addresses, etc.).
       *          For all these reasons the stream might become unstable and thus not
       *          usable anymore.
       *          Calling this function allows to determine whether this is the case
       *          or if the stream is still valid. Note that usually if this method
       *          returns `false` the stream should be recreated as most (if not all)
       *          of subsequent calls will return an error.
       * @param stream - the cuda stream to check for validity.
       * @return - `true` if no errors where reported for this stream (and thus some
       *           other operations can be scheduled) and `false` otherwise.
       */
      bool
      check(cuda::stream_t stream);

      /**
       * @brief - Used to wait on the input stream that it completes its task. Note
       *          that this process will block the calling thread until the stream
       *          returns from its computation.
       * @param stream - the stream to wait for.
       * @return - `true` if the stream successfully terminated and `false` if some
       *           errors were detected.
       */
      bool
      wait(cuda::stream_t stream);

      /**
       * @brief - Used to perform tthe destruction of the resources occupied by the
       *          input stream. This should typically be called when this stream is
       *          not needed anymore because all relevant operations have already
       *          been scheduled.
       * @param stream - the stream to destroy.
       * @return - `true` if the stream was successfully destroyed and `false` if
       *           some error occurred (in which case one can check `getLastError`).
       */
      bool
      destroy(cuda::stream_t stream);

      /**
       * @brief - Return the last error registered by the wrapper. Note that it might
       *          not be directly linked to the last `launch` call.
       *          Any subsequent call to this method will return the empty string (so
       *          save the return value when needed).
       * @return - the last error description that was reported by the API.
       */
      std::string
      getLastError();

    private:

      /**
       * @brief - Used to check whether the error code provided as input yields an
       *          error.
       * @param error - the error code to check.
       * @return - `true` if the error code indicates an error and `false` otherwise.
       */
      bool
      isError(cudaError_t error);

    private:

      /**
       * @brief - Protects this object from concurrent accesses.
       */
      std::mutex m_propsLocker;

      /**
       * @brief - A string representing the last error that occurred during an API call.
       *          This value is reset whenever the `getLastError` method is called.
       */
      std::string m_lastError;
  };

}

# include "CudaWrapper.hxx"

#endif    /* CUDA_WRAPPER_CUH */
