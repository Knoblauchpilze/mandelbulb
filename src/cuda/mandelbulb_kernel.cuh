#ifndef    MANDELBULB_KERNEL_CUH
# define   MANDELBULB_KERNEL_CUH

# include <cstdint>
# include <cuda_runtime.h>

/**
 * @brief - Define the number of lights available for shading.
 */
# define MAX_LIGHTS 5

namespace mandelbulb {

  namespace gpu {

    /**
     * @brief - This enumeration describe the possible values for accessing
     *          values in the lights array.
     */
    enum LightProp {
      ACTIVE  = 0,

      DIR_X   = 1,
      DIR_Y   = 2,
      DIR_Z   = 3,

      COLOR_R = 4,
      COLOR_G = 5,
      COLOR_B = 6,

      INTENSITY = 7,

      COUNT = 8
    };

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

      // Fractal color.
      float f_r;
      float f_g;
      float f_b;

      // No data color.
      float no_data_r;
      float no_data_g;
      float no_data_b;

      // Lights.
      float blending;
      float lights[MAX_LIGHTS * COUNT];

      // Tonemap.
      float exposure;
      float burnout;
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

/**
 * @brief - Convenience macros to access some properties of lights within the
 *          `KernelProps` object.
 */
# define LIGHT_PROP(DATA, ID, PROP) DATA[ID * gpu::COUNT + PROP]

#endif    /* MANDELBULB_KERNEL_CUH */
