
# include "CudaWrapper.cuh"

namespace utils {

  CudaWrapper::CudaWrapper():
    CoreObject(std::string("wrapper")),

    m_propsLocker(),
    m_lastError()
  {
    setService("cuda");
  }

  bool
  CudaWrapper::launch(cuda::stream_t stream,
                      std::function<cudaError_t(void)> func)
  {
    // Perform the call to the function.
    cudaError_t err = func();

    if (isError(err)) {
      const std::lock_guard guard(m_propsLocker);
      m_lastError = cudaGetErrorString(err);

      return false;
    }

    // Wait for the stream to finish its task.
    return wait(stream);
  }

  cuda::stream_t
  CudaWrapper::create(bool* success) {
    // Attempt to create the stream.
    cudaStream_t rawStream;

    cudaError_t err = cudaStreamCreateWithFlags(&rawStream, cudaStreamNonBlocking);

    checkAndSaveError(err);

    // Fill the error status.
    if (success != nullptr) {
      *success = !isError(err);
    }

    return reinterpret_cast<cuda::stream_t>(rawStream);
  }

  bool
  CudaWrapper::wait(cuda::stream_t stream) {
    // Cast the input stream to usable data.
    cudaStream_t rawStream = reinterpret_cast<cudaStream_t>(stream);

    // Wait for the stream to terminate.
    cudaError_t err = cudaStreamSynchronize(rawStream);

    checkAndSaveError(err);

    return !isError(err);
  }

  bool
  CudaWrapper::destroy(cuda::stream_t stream) {
    // Attempt to destroy the stream.
    cudaStream_t rawStream = reinterpret_cast<cudaStream_t>(stream);

    cudaError_t err = cudaStreamDestroy(rawStream);

    checkAndSaveError(err);

    if (!isError(err)) {
      verbose("Destroyed cuda stream");
    }

    return !isError(err);
  }

  void*
  CudaWrapper::allocate(unsigned size,
                        bool* success)
  {
    // Declare the output memory location.
    void* out = nullptr;

    // Perform the allocation.
    cudaError_t err = cudaMalloc(&out, size);

    checkAndSaveError(err);

    // Populate `success` boolean if needed.
    if (success != nullptr) {
      *success = !isError(err);
    }

    return out;
  }

  void*
  CudaWrapper::allocate2D(const utils::Sizei& size,
                          unsigned elemSize,
                          bool* success)
  {
    // Declare the output buffer.
    void* out = nullptr;

    // Compute aligned dimensions of the buffer.
    int wBytes = size.w() * elemSize;
    int h = size.h();

    // Perform the allocation.
    cudaError_t err = cudaMalloc(&out, wBytes * h);

    checkAndSaveError(err);

    // Populate `success` boolean if needed.
    if (success != nullptr) {
      *success = !isError(err);
    }

    return out;
  }

  bool
  CudaWrapper::free(void* buffer) {
    // In case the buffer is already `null` we don't need to do anything.
    if (buffer == nullptr) {
      warn("Could not release gpu memory, buffer is already null");
      return true;
    }

    // Release the memory.
    cudaError_t err = cudaFree(buffer);

    checkAndSaveError(err);

    return !isError(err);
  }

  bool
  CudaWrapper::copyToDevice(cuda::stream_t stream,
                            void* src,
                            unsigned elemSize,
                            void* dst)
  {
    // Cast input stream to usable data.
    cudaStream_t rawStream = reinterpret_cast<cudaStream_t>(stream);

    // Check consistency.
    if (src == nullptr) {
      error(
        std::string("Could not perform copy of data with size ") + std::to_string(elemSize) + " to host",
        std::string("Invalid null source pointer")
      );
    }
    if (dst == nullptr) {
      error(
        std::string("Could not perform copy of data with size ") + std::to_string(elemSize) + " to host",
        std::string("Invalid null destination pointer")
      );
    }

    cudaError_t err = cudaMemcpyAsync(dst, src, elemSize, cudaMemcpyHostToDevice, rawStream);

    checkAndSaveError(err);

    return !isError(err);
  }

  bool
  CudaWrapper::copyToDevice2D(cuda::stream_t stream,
                              const utils::Sizei& size,
                              unsigned elemSize,
                              void* src,
                              void* dst)
  {
    // Cast input stream to usable data.
    cudaStream_t rawStream = reinterpret_cast<cudaStream_t>(stream);

    // Check consistency.
    if (src == nullptr) {
      error(
        std::string("Could not perform copy of data with size ") + size.toString() + " to device",
        std::string("Invalid null source pointer")
      );
    }
    if (dst == nullptr) {
      error(
        std::string("Could not perform copy of data with size ") + size.toString() + " to device",
        std::string("Invalid null destination pointer")
      );
    }

    // Perform the copy.
    cudaError_t err = cudaMemcpy2DAsync(
      dst,
      size.w() * elemSize,
      src,
      size.w() * elemSize,
      size.w() * elemSize,
      size.h(),
      cudaMemcpyHostToDevice,
      rawStream
    );

    checkAndSaveError(err);

    return !isError(err);
  }

  bool
  CudaWrapper::copyToHost(cuda::stream_t stream,
                          void* src,
                          unsigned elemSize,
                          void* dst)
  {
    // Cast input stream to usable data.
    cudaStream_t rawStream = reinterpret_cast<cudaStream_t>(stream);

    // Check consistency.
    if (src == nullptr) {
      error(
        std::string("Could not perform copy of data with size ") + std::to_string(elemSize) + " to host",
        std::string("Invalid null source pointer")
      );
    }
    if (dst == nullptr) {
      error(
        std::string("Could not perform copy of data with size ") + std::to_string(elemSize) + " to host",
        std::string("Invalid null destination pointer")
      );
    }

    cudaError_t err = cudaMemcpyAsync(dst, src, elemSize, cudaMemcpyDeviceToHost, rawStream);

    checkAndSaveError(err);

    return !isError(err);
  }

  bool
  CudaWrapper::copyToHost2D(cuda::stream_t stream,
                            const utils::Sizei& size,
                            unsigned elemSize,
                            void* src,
                            void* dst)
  {
    // Cast input stream to usable data.
    cudaStream_t rawStream = reinterpret_cast<cudaStream_t>(stream);

    // Check consistency.
    if (src == nullptr) {
      error(
        std::string("Could not perform copy of data with size ") + size.toString() + " to host",
        std::string("Invalid null source pointer")
      );
    }
    if (dst == nullptr) {
      error(
        std::string("Could not perform copy of data with size ") + size.toString() + " to host",
        std::string("Invalid null destination pointer")
      );
    }

    // Perform the copy.
    cudaError_t err = cudaMemcpy2DAsync(
      dst,
      size.w() * elemSize,
      src,
      size.w() * elemSize,
      size.w() * elemSize,
      size.h(),
      cudaMemcpyDeviceToHost,
      rawStream
    );

    checkAndSaveError(err);

    return !isError(err);
  }

}
