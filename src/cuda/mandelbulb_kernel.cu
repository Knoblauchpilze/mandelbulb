
# include "mandelbulb_kernel.cuh"
# include <cuda_profiler_api.h>

# include <stdio.h>

namespace {

  __device__ __inline__
  float3&
  operator+=(float3& lhs, const float3& rhs) {
    lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z;

    return lhs;
  }

  __device__ __inline__
  float4&
  operator+=(float4& lhs, const float4& rhs) {
    lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z;

    return lhs;
  }

  __device__ __inline__
  float4
  operator+(const float4& lhs, const float4& rhs) {
    return make_float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, 0.0f);
  }

  __device__ __inline__
  float4
  operator-(const float4& v) {
    return make_float4(-v.x, -v.y, -v.z, 0.0f);
  }

  __device__ __inline__
  float3
  operator*(const float3& lhs, float v) {
    return make_float3(lhs.x * v, lhs.y * v, lhs.z * v);
  }

  __device__ __inline__
  float3
  operator*(float v, const float3& lhs) {
    return lhs * v;
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
  dot(const float4& lhs, const float4& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
  }

}

namespace mandelbulb {
  namespace gpu {

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

        // We could add a test like `if (r < bailout)` here but we won't for two
        // reasons:
        // - first it only saves a single iteration. Chances are that if this it
        //   is already larger than the bailout things will get worse.
        // - second it allows to actually initialize a correct value for the `dr`
        //   value which in turns allow to produce a valid distance estimation
        //   even for the cases where the point is outside of the convergence
        //   radius from the beginning. This helps a *lot* for the raymarching.

        // Convert to spherical coordinates.
        if (r < bailout) {
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

      // Return the distance estimator. Note that compared to the formular described here:
      // http://celarek.at/wp/wp-content/uploads/2014/05/realTimeFractalsReport.pdf
      // we ivided itby `2`. Another discussion here:
      // http://www.fractalforums.com/3d-fractal-generation/true-3d-mandlebrot-type-fractal/msg8540/#msg8540
      // It seems that the distance estimator is very inaccurate anyways for small values
      // of the bailout. So we need to lower it to not pass through the fractal. This is
      // basically what we had to do by dividing it by `2`. As it's an estimation anyways
      // we will consider that it's okay.
      return 0.25f * logf(r) * r / dr;
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

    __device__ __inline__
    float3
    directional_light(const float4& p,
                      const float4& n,
                      const float4& light,
                      const float3& color,
                      float proxThresh,
                      unsigned maxSteps,
                      unsigned acc,
                      float bailout,
                      float exp)
    {
      // Compute lighting for this point. We first need to determine whether
      // we can reach the light from the point. We will raymarch the ray to
      // the direction of the light and see whether we hit the fractal at
      // some point before reaching the bailout distance.
      float tDist = 0.0f;
      uint32_t steps = 0u;
      float dist = proxThresh + 1.0f;
      bool escaped = false;

      // Offset a bit the starting point to not get stuck in the
      // fractal: indeed `p` is supposed to be part of the fractal.
      float4 s = p + proxThresh * n;
      float4 phot = s;

      while (steps < maxSteps && dist > proxThresh / 2.0f && !escaped) {
        // Get an estimation of the distance.
        dist = get_distance_estimate(phot, acc, bailout, exp);

        // Add this and move on to the next step.
        tDist += dist;
        ++steps;

        // March on the ray.
        phot = s + tDist * light;

        // Update whether we escaped: as the light is supposed to be
        // infinitely far away we just need to check whether we are
        // farther than the bailout radius. If this is the case it
        // means that the fractal cannot be intersected anymore and
        // that we will be able to reach the light.
        escaped = (length(phot) >= bailout);
      }

      // Always add an ambient component to the color but only adjust for
      // the final color if we can reach a light from this point.
      if (escaped) {
        return color * fminf(1.0f, fmaxf(0.0f, dot(light, n)));
      }

      return make_float3(0.0f, 0.0f, 0.0f);
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

      // Retrieve global memory elements to local registries.
      uint32_t acc = props->accuracy;
      float exp = props->exponent;
      float bailout = props->bailout;
      float proxThresh = props->hit_thresh;
      uint32_t maxSteps = props->ray_steps;

      float4 e = make_float4(props->eye_x, props->eye_y, props->eye_z, 0.0f);

      // Generate the ray direction.
      float4 dir;
      {
        // Convert input properties into usable data.
        float4 u = make_float4(props->u_x, props->u_y, props->u_z, 0.0f);
        float4 v = make_float4(props->v_x, props->v_y, props->v_z, 0.0f);
        float4 w = make_float4(props->w_x, props->w_y, props->w_z, 0.0f);

        float4 dims = make_float4(
          1.0f * props->min_x,
          1.0f * props->min_y,
          1.0f * props->tot_w,
          1.0f * props->tot_h
        );

        dir = generate_ray_dir(x, y, u, v, w, dims);
      }


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
        res[4u * (y * width + x)] = -1.0f;

        // Update with distance to the fractal object if we reached
        // it.
        if (!escaped) {
          res[4u * (y * width + x)] = tDist;
        }
      }
    }

    __global__
    void
    mandelbulb_lighting(const void* data,
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

      // Prevent execution of out of bounds rays.
      if (y >= height || x >= width) {
        return;
      }

      // Do not handle rays that did not intersect the fractal object.
      if (res[4u * (y * width + x)] < 0.0f) {
        // Set the no data color.
        res[4u * (y * width + x) + 1u] = props->no_data_r;
        res[4u * (y * width + x) + 2u] = props->no_data_g;
        res[4u * (y * width + x) + 3u] = props->no_data_b;

        return;
      }

      // Retrieve global memory elements to local registries.
      uint32_t acc = props->accuracy;
      float exp = props->exponent;
      float bailout = props->bailout;
      float proxThresh = props->hit_thresh;
# ifndef NORMAL_DISPLAY
      uint32_t maxSteps = props->ray_steps;
# endif

      float4 e = make_float4(props->eye_x, props->eye_y, props->eye_z, 0.0f);

      // Retrieve the distance estimated for the fractal point.
      float depth = res[4u * (y * width + x)];

      // Compute the current point to light.
      float4 dir;
      {
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
        dir = generate_ray_dir(x, y, u, v, w, dims);
      }
      float4 p = e + depth * dir;

      // Compute a local normal by estimating the distance to the fractal
      // on neighboring points.
      float eps = proxThresh / 2.0f;

      float dp = get_distance_estimate(p, acc, bailout, exp);
      float dx = get_distance_estimate(p + make_float4(eps, 0.0f, 0.0f, 0.0f), acc, bailout, exp) - dp;
      float dy = get_distance_estimate(p + make_float4(0.0f, eps, 0.0f, 0.0f), acc, bailout, exp) - dp;
      float dz = get_distance_estimate(p + make_float4(0.0f, 0.0f, eps, 0.0f), acc, bailout, exp) - dp;

      float4 n = normalize(make_float4(dx, dy, dz, 0.0f));

      // Compute the color by lighting the object with provided information
      // from the kernel properties. We will cycle through lights and only
      // consider the active one to provide lighting.
      float3 c = make_float3(0.0f, 0.0f, 0.0f);

      for (unsigned id = 0u ; id < MAX_LIGHTS ; ++id) {
        // Retrieve light properties and use it if it is active.
        if (LIGHT_PROP(props->lights, id, ACTIVE) < 0.0f) {
          continue;
        }

        float4 l = normalize(
          make_float4(
            LIGHT_PROP(props->lights, id, DIR_X),
            LIGHT_PROP(props->lights, id, DIR_Y),
            LIGHT_PROP(props->lights, id, DIR_Z),
            0.0f
          )
        );

        float3 lc = make_float3(
          LIGHT_PROP(props->lights, id, COLOR_R),
          LIGHT_PROP(props->lights, id, COLOR_G),
          LIGHT_PROP(props->lights, id, COLOR_B)
        );

        float w = LIGHT_PROP(props->lights, id, INTENSITY);

        c += w * directional_light(p, n, -l, lc, proxThresh, maxSteps, acc, bailout, exp);
      }

      // Apply a simple reinhard tonemapping to handle bruned areas.
      float lum = c.x * 0.2126f + c.y * 0.7152f + c.z * 0.0722f;

      float mapped = lum * props->exposure;
      mapped *= (1.0f + mapped * props->burnout) / (1.0f + mapped);

      float tone = mapped / lum;

      // Save the result in the output buffer.
      res[4u * (y * width + x) + 1u] = tone * c.x;
      res[4u * (y * width + x) + 2u] = tone * c.y;
      res[4u * (y * width + x) + 3u] = tone * c.z;
    }

  }

  cudaError_t
  mandelbulb_kernel_wrapper(cudaStream_t stream,
                            const void* data,
                            void* out,
                            unsigned w,
                            unsigned h)
  {
    // Compute the grid size from the input dimensions. We will compute secondary rays
    static const unsigned tpb_x = 32u;
    static const unsigned tpb_y = 8u;

    dim3 grid((w + tpb_x - 1) / tpb_x, (h + tpb_y - 1) / tpb_y);
    dim3 block(tpb_x, tpb_y);

    cudaProfilerStart();

    // Launch the kernel.
    gpu::mandelbulb_kernel<<<grid, block, 0, stream>>>(data, out, w, h);

    // Peek at any kernel launch failure.
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
      return err;
    }

    // Perform lighting.
    gpu::mandelbulb_lighting<<<grid, block, 0, stream>>>(data, out, w, h);

    cudaProfilerStop();

    // Peek at any kernel launch failure.
    return cudaPeekAtLastError();
  }

}
