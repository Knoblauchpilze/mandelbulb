
# include "CudaWrapper.cuh"

namespace utils {

  CudaWrapper::CudaWrapper(const utils::Vector2i& alignment):
    CoreObject(std::string("wrapper")),

    m_propsLocker(),
    m_lastError(),

    m_alignment(alignment)
  {
    setService("cuda");

    // Consistency check.
    if (m_alignment.x() < 0 || m_alignment.y() < 0) {
      error(
        std::string("Cannot create cuda wrapper"),
        std::string("Invalid alignment pattern provided: ") + m_alignment.toString()
      );
    }
  }

  bool
  CudaWrapper::launch(cuda::stream_t stream,
                      std::function<cudaError_t(void)> func)
  {
    // Perform the call to the function.
    cudaError_t err = func();

    if (isError(err)) {
      Guard guard(m_propsLocker);
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
      log("Destroyed cuda stream", utils::Level::Debug);
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
                          unsigned& step,
                          bool* success)
  {
    // Declare the output buffer.
    void* out = nullptr;

    // Compute aligned dimensions of the buffer.
    int wBytes = size.w() * elemSize;
    int h = size.h();

    if (m_alignment.x() > 0) {
      wBytes += (m_alignment.x() - wBytes % m_alignment.x());
    }
    if (m_alignment.y() > 0) {
      h += (m_alignment.y() - h % m_alignment.y());
    }

    // Perform the allocation.
    cudaError_t err = cudaMalloc(&out, wBytes * h);

    checkAndSaveError(err);

    // Populate `success` boolean if needed.
    if (success != nullptr) {
      *success = !isError(err);
    }

    // Populate the step if the allocation was successful.
    if (!isError(err)) {
      step = wBytes;
    }

    return out;
  }

  bool
  CudaWrapper::free(void* buffer) {
    // In case the buffer is already `null` we don't need to do anything.
    if (buffer == nullptr) {
      log(
        std::string("Could not release gpu memory, buffer is already null"),
        utils::Level::Error
      );

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
                              void* src,
                              unsigned srcStep,
                              void* dst,
                              unsigned dstStep)
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
      dstStep,
      src,
      srcStep,
      size.w(),
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
                            void* src,
                            unsigned srcStep,
                            void* dst,
                            unsigned dstStep)
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
      dstStep,
      src,
      srcStep,
      size.w(),
      size.h(),
      cudaMemcpyDeviceToHost,
      rawStream
    );

    checkAndSaveError(err);

    return !isError(err);
  }

}
