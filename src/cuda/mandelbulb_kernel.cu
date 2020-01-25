
# include "mandelbulb_kernel.cuh"

# include <stdio.h>

namespace mandelbulb {
  namespace gpu {

    __device__ __inline__
    float4&
    operator+=(float4& lhs, const float4& rhs) {
      lhs.x += rhs.x;
      lhs.y += rhs.y;
      lhs.z += rhs.z;

      return lhs;
    }

    __device__ __inline__
    float4
    operator+(const float4& lhs, const float4& rhs) {
      return make_float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, 0.0f);
    }

    __device__ __inline__
    float4
    operator*(const float4& lhs, float v) {
      return make_float4(lhs.x * v, lhs.y * v, lhs.z * v, 0.0f);
    }

    __device__ __inline__
    float4
    operator*(float v, const float4& lhs) {
      return lhs * v;
    }

    __device__ __inline__
    float
    length(const float4& v) {
      return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    __device__ __inline__
    float4
    normalize(const float4& v) {
      float l = length(v);
      return make_float4(v.x / l, v.y / l, v.z / l, 0.0f);
    }

    __device__ __inline__
    float
    get_distance_estimate(const float4& p,
                          uint32_t acc,
                          float bailout,
                          float exponent)
    {
      // Register loop variables.
      uint32_t iter = 0u;
      float4 z = p;

      float r = bailout - 1.0f;
      float dr = 1.0f;
      float theta, phi;
      float zr;

      // Compute as many iterations as needed. We stop either because we reached
      // the input accuracy or because the point has a greater magnitude to be
      // considered not converging.
      while (iter < acc && r < bailout) {
        r = length(z);

        // Only iterate if we didn't escape yet.
        if (r < bailout) {
          // Convert to spherical coordinates.
          theta = acosf(z.z / r);
          phi = atan2f(z.y, z.x);

          // Update distance estimator.
          dr = powf(r, exponent - 1.0f) * exponent * dr + 1.0f;

          // Scale and rotate the point.
          zr = pow(r, exponent);
          theta *= exponent;
          phi *= exponent;

          // Convert back to cartesian coordinates.
          z.x = zr * cosf(phi) * sinf(theta);
          z.y = zr * sinf(phi) * sinf(theta);
          z.z = zr * cosf(theta);

          z += p;
        }

        ++iter;
      }

      // Return the distance estimator.
      return 0.5f * logf(r) * r / dr;
    }

    __device__ __inline__
    float4
    generate_ray_dir(int x,
                     int y,
                     const float4& u,
                     const float4& v,
                     const float4& w,
                     const float4& dims)
    {
      // Express this pixel's value into a percentage of the total
      // area and offset it to obtain a centered value.
      float percX = -0.5f + 1.0f * (x + dims.x + dims.z / 2) / dims.z;
      float percY = -0.5f + 1.0f * (y + dims.y + dims.w / 2) / dims.w;

      // This can be used to compute the ray direction.
      return normalize(percX * u + percY * v + w);
    }

    __global__
    void
    mandelbulb_kernel(const void* data,
                      void* out,
                      unsigned width,
                      unsigned height)
    {
      // Cast the input data to valid properties.
      const KernelProps* props = reinterpret_cast<const KernelProps*>(data);
      float* res = reinterpret_cast<float*>(out);

      // Compute the position of this thread: this will determine the
      // ray to be processed.
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;

      // Convert input properties into usable data.
      float4 e = make_float4(props->eye_x, props->eye_y, props->eye_z, 0.0f);
      float4 u = make_float4(props->u_x, props->u_y, props->u_z, 0.0f);
      float4 v = make_float4(props->v_x, props->v_y, props->v_z, 0.0f);
      float4 w = make_float4(props->w_x, props->w_y, props->w_z, 0.0f);

      float4 dims = make_float4(
        1.0f * props->min_x,
        1.0f * props->min_y,
        1.0f * props->tot_w,
        1.0f * props->tot_h
      );

      // Generate the ray direction.
      float4 dir = generate_ray_dir(x, y, u, v, w, dims);

      // Retrieve global memory elements to local registries.
      float proxThresh = props->hit_thresh;
      uint32_t maxSteps = props->ray_steps;
      float bailout = props->bailout;
      float exp = props->exponent;
      uint32_t acc = props->accuracy;

      // March on the ray until we reach a point close enough to
      // the fractal or we escape or we reach the maximum number
      // of iterations allowed for a ray.
      float tDist = 0.0f;
      uint32_t steps = 0u;
      float dist = proxThresh + 1.0f;
      bool escaped = false;

      float4 p = e + tDist * dir;

      while (steps < maxSteps && dist > proxThresh && !escaped) {
        // Get an estimation of the distance.
        dist = get_distance_estimate(p, acc, bailout, exp);

        // Add this and move on to the next step.
        tDist += dist;
        ++steps;

        // March on the ray.
        p = e + tDist * dir;

        // Update escape status.
        escaped = (length(p) >= bailout);
      }

      // Save the result to the output buffer. We need to prevent
      // writing of elements which are out of bounds.
      if (y < height && x < width) {
        // Initialize with invalid data.
        res[y * width + x] = -1.0f;

        // Update with distance to the fractal object if we reached
        // it.
        if (!escaped) {
          res[y * width + x] = tDist;
        }
      }
    }

  }

  cudaError_t
  mandelbulb_kernel_wrapper(cudaStream_t stream,
                            const void* data,
                            void* out,
                            unsigned w,
                            unsigned h)
  {
    // Compute the grid size from the input dimensions.
    static const unsigned tpb_x = 32u;
    static const unsigned tpb_y = 8u;

    dim3 grid((w + tpb_x - 1) / tpb_x, (h + tpb_y - 1) / tpb_y);
    dim3 block(tpb_x, tpb_y);

    // Launch the kernel.
    gpu::mandelbulb_kernel<<<grid, block, 0, stream>>>(data, out, w, h);

    // Peek at any kernel launch failure.
    return cudaPeekAtLastError();
  }

}
