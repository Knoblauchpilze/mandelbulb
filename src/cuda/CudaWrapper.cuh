#ifndef    CUDA_WRAPPER_CUH
# define   CUDA_WRAPPER_CUH

# include <mutex>
# include <vector>
# include <functional>
# include <cuda_runtime.h>
# include <core_utils/CoreObject.hh>
# include <maths_utils/Vector2.hh>
# include <maths_utils/Size.hh>

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

      /**
       * @brief - Used to perform the allocation of a segment of memory with said size
       *          expressed in bytes.
       * @param size - the size of the memory to allocate in bytes.
       * @param success - a value set to `true` in case the allocation succeeeded (and
       *                  if it is not `null`).
       * @return - a pointer to the device memory allocated or `null` if the memory is
       *           not valid.
       */
      void*
      allocate(unsigned size,
               bool* success);

      /**
       * @brief - Used to perform the allocation of a patch of memory allowing to hold
       *          `size` elements of size `elemSize` byte(s).
       *          The memory is allocated using some padding so as to keep the length
       *          of each individual line of data an entire multiple of some dimension
       *          provided when building this object.
       * @param size - the size of the memory to allocate.
       * @param elemSize - the size in byte(s) of each inidividual elements to store.
       *                   This value is typically obtained through a call to `sizeof`
       *                   on the relevant data type.
       * @param success - a value set to `true` in case the allocation succeeded (and
       *                  if it is not `null`).
       * @return - a pointer to the device memory allocated by this function or `null`
       *           if something went wrong.
       */
      void*
      allocate2D(const Sizei& size,
                 unsigned elemSize,
                 bool* success);

      /**
       * @brief - Call the underlying cuda API to release the resources pointed at by
       *          the input `buffer`. This pointer should no longer be used after the
       *          free operation.
       *          Nothing happen if the buffer is `null`.
       * @param buffer - the gpu memory to release.
       * @return - `true` if the pointer was successfully released or `false` otherwise.
       */
      bool
      free(void* buffer);

      /**
       * @brief - Used to perform the copy of the single value from host memory to device
       *          memory. If any of the `src` or `dst` pointers are `null` an error is
       *          raised.
       *          Note that we assume that the `dst` is actually large enough to handle
       *          and receive `elemSize` elements.
       * @param stream - the stream to use to copy the data.
       * @param src - the host pointer to copy.
       * @param elemSize - the size in bytes of the data to copy.
       * @param dst - the device pointer to copy to.
       * @return - `true` if the copy succeeded and `false` otherwise.
       */
      bool
      copyToDevice(cuda::stream_t stream,
                   void* src,
                   unsigned elemSize,
                   void* dst);

      /**
       * @brief - Used to perform the copy to device memory of the input host pointer.
       *          The provided stream is used in order to schedule the copy along the
       *          other pending gpu operations.
       *          Note that in case any of `src` or `dst` are `null` an error is raised.
       * @param size - the size of the input data to copy.
       * @param elemSize - the size in bytes of a single element of the `src` buffer.
       * @param src - the host pointer to copy.
       * @param dst - the device pointer to which the data should be copied.
       * @return - `true` if the copy could be performed and `false` otherwise.
       */
      bool
      copyToDevice2D(cuda::stream_t stream,
                     const utils::Sizei& size,
                     unsigned elemSize,
                     void* src,
                     void* dst);

      /**
       * @brief - Used to perform the copy of the single value from device memory to host
       *          memory. If any of the `src` or `dst` pointers are `null` an error is
       *          raised.
       *          Note that we assume that the `dst` is actually large enough to handle
       *          and receive `elemSize` elements.
       * @param stream - the stream to use to copy the data.
       * @param src - the device pointer to copy.
       * @param elemSize - the size in bytes of the data to copy.
       * @param dst - the host pointer to copy to.
       * @return - `true` if the copy succeeded and `false` otherwise.
       */
      bool
      copyToHost(cuda::stream_t stream,
                 void* src,
                 unsigned elemSize,
                 void* dst);

      /**
       * @brief - Used to perform the copy from device memory of the input device data.
       *          This is almost the exact opposite of the `copyToDevice2D` method.
       *          Note that in case any of `src` or `dst` are `null` an error is raised.
       * @param size - the size of the device data to copy.
       * @param elemSize - the size in bytes of a single element of the `src` buffer.
       * @param src - the device pointer to copy.
       * @param dst - the host pointer to which the data should be copied.
       * @return - `true` if the copy could be performed and `false` otherwise.
       */
      bool
      copyToHost2D(cuda::stream_t stream,
                   const utils::Sizei& size,
                   unsigned elemSize,
                   void* src,
                   void* dst);

    private:

      /**
       * @brief - Used to check whether the error code provided as input yields an
       *          error.
       * @param error - the error code to check.
       * @return - `true` if the error code indicates an error and `false` otherwise.
       */
      bool
      isError(const cudaError_t& error);

      /**
       * @brief - Used to update the internal `m_lastError` string in case the input
       *          `error` value indicates an error. The string saved corresponds to
       *          the description provided by the cuda API of the error code.
       * @param error - the error which should be saved in case it indicates a real
       *                problem.
       */
      void
      checkAndSaveError(const cudaError_t& error);

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
