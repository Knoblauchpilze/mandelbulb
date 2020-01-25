
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

    if (isError(err)) {
      Guard guard(m_propsLocker);
      m_lastError = cudaGetErrorString(err);
    }

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

    if (isError(err)) {
      Guard guard(m_propsLocker);
      m_lastError = cudaGetErrorString(err);
    }

    return !isError(err);
  }

  bool
  CudaWrapper::destroy(cuda::stream_t stream) {
    // Attempt to destroy the stream.
    cudaStream_t rawStream = reinterpret_cast<cudaStream_t>(stream);

    cudaError_t err = cudaStreamDestroy(rawStream);

    if (isError(err)) {
      Guard guard(m_propsLocker);
      m_lastError = cudaGetErrorString(err);
    }

    if (!isError(err)) {
      log("Destroyed cuda stream", utils::Level::Debug);
    }

    return !isError(err);
  }

}
