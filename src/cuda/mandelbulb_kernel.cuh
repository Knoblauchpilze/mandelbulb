#ifndef    MANDELBULB_KERNEL_CUH
# define   MANDELBULB_KERNEL_CUH

# include <cstdint>
# include <cuda_runtime.h>

namespace mandelbulb {

  namespace gpu {

    /**
     * @brief - Convenience define to reference all the parameters needed by
     *          the mandelbulb computation kernel.
     */
    struct KernelProps {
      // accuracy.
      uint32_t accuracy;
      // exponent.
      float exponent;
      // bailout.
      float bailout;
      // hit threshold.
      float hit_thresh;
      // ray steps.
      uint32_t ray_steps;

      // `eye` vector.
      float eye_x;
      float eye_y;
      float eye_z;
      // `u` vector.
      float u_x;
      float u_y;
      float u_z;
      // `v` vector.
      float v_x;
      float v_y;
      float v_z;
      // `w` vector.
      float w_x;
      float w_y;
      float w_z;

      // min `x`.
      int min_x;
      // min `y`.
      int min_y;
      // total `w`.
      int tot_w;
      // total `h`.
      int tot_h;
    };

  }

  /**
   * @brief - Used to compute the distance to the mandelbulb fractal using the
   *          arguments provided as properties. The first argument is assumed
   *          to represent a `KernelProps` object.
   *          The results are saved in the `out` buffer assumed to be a vector
   *          of floats where the depth is saved.
   * @param stream - the stream to use to schedule the kernel launch.
   * @param data - a raw pointer on device memory holding the properties of the
   *               fractal to render (accuracy, camera settings, etc.).
   * @param out - the output buffer where the distance should be saved.
   * @param w - the width of the kernel to launch. Represents the width of the
   *            `out` buffer.
   * @param h - the height of the kernel to launch. Represents the height of the
   *            `out` buffer.
   * @return - any error or `cudaSuccess` if the kernel successfully executed.
   */
  cudaError_t
  mandelbulb_kernel_wrapper(cudaStream_t stream,
                            const void* data,
                            void* out,
                            unsigned w,
                            unsigned h);

}

#endif    /* MANDELBULB_KERNEL_CUH */
