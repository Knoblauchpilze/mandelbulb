
# include <cuda_runtime.h>
# include <core_utils/CoreException.hh>
# include <stdio.h>
# include <iostream>

namespace mandelbulb {
  namespace gpu {

    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    __host__
    inline
    void
    cuda_check_launch(cudaError_t code,
                      const std::string& name)
    {
      if (code != cudaSuccess) {
        throw utils::CoreException(
          std::string("Error while launching kernel \"") + name + "\"",
          std::string("launch"),
          std::string("cuda"),
          std::string("Launch returned ") + std::to_string(code) + " (msg: \"" + cudaGetErrorString(code) + "\""
        );
      }
    }

    __device__
    void
    print_kernel(unsigned x, unsigned y) {
      printf("[%dx%d] Kernel says hello !\n", x, y);
    }

    __global__
    void
    test_kernel() {
      unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

      print_kernel(x, y);
    }

    __host__
    void
    test_kernel_wrapper() {
      dim3 grid = dim3(2, 3, 1);
      dim3 block = dim3(4, 5);

      std::cout << "[TEST] Executing kernel wrapper" << std::endl;

      test_kernel<<<grid, block>>>();

      cuda_check_launch(cudaPeekAtLastError(), "test_kernel(launch)");
      cuda_check_launch(cudaDeviceSynchronize(), "test_kernel(execute)");
    }

  }
}
