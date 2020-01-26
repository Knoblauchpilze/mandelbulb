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
      // General rendering properties.
      uint32_t accuracy;
      float exponent;
      float bailout;
      float hit_thresh;
      uint32_t ray_steps;

      // Camera properties.
      float eye_x;
      float eye_y;
      float eye_z;

      float u_x;
      float u_y;
      float u_z;

      float v_x;
      float v_y;
      float v_z;

      float w_x;
      float w_y;
      float w_z;

      // Area description.
      int min_x;
      int min_y;
      int tot_w;
      int tot_h;

      // No data color.
      float no_data_r;
      float no_data_g;
      float no_data_b;

      // Light 1.
      bool l1_active;

      float l1_dx;
      float l1_dy;
      float l1_dz;

      float l1_r;
      float l1_g;
      float l1_b;

      float l1_i;

      // Light 2.
      bool l2_active;

      float l2_dx;
      float l2_dy;
      float l2_dz;

      float l2_r;
      float l2_g;
      float l2_b;

      float l2_i;

      // Light 3.
      bool l3_active;

      float l3_dx;
      float l3_dy;
      float l3_dz;

      float l3_r;
      float l3_g;
      float l3_b;

      float l3_i;
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
