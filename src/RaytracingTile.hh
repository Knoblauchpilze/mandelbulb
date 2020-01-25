#ifndef    RAYTRACING_TILE_HH
# define   RAYTRACING_TILE_HH

# include <mutex>
# include <vector>
# include <memory>
# include <maths_utils/Box.hh>
# include <maths_utils/Vector3.hh>
# include "CudaJob.hh"
# include "RenderProperties.hh"
# include "mandelbulb_kernel.cuh"

namespace mandelbulb {

  class RaytracingTile: public utils::CudaJob {
    public:

      /**
       * @brief - Associates this raytracing tile with the specified
       *          camera and an area which should represent a part of
       *          the camera plane.
       *          An error is raised if either the `camera` or the
       *          `area` are invalid.
       * @param eye - the position of the origin of the camera plane.
       * @param u - the local `x` axis in the camera plane.
       * @param v - the local `y` axis in the camera plane.
       * @param w - a vector perpendicular to the camera plane.
       * @param total - the total area into which this tile is inscribed.
       * @param area - the area of the camera plane which should be
       *               processed by this tile.
       * @param props - the rendering properties to use to perform the
       *                computations (i.e. description of the fractal).
       */
      RaytracingTile(const utils::Vector3f& eye,
                     const utils::Vector3f& u,
                     const utils::Vector3f& v,
                     const utils::Vector3f& w,
                     const utils::Sizei& total,
                     const utils::Boxi& area,
                     const RenderProperties& props);

      virtual ~RaytracingTile() = default;

      /**
       * @brief - Provide the size in bytes of the input parameters to provide so
       *          that this kind of job can be executed.
       *          In the case of a raytracing tile we need to provide some info
       *          about the properties to compute the fractal. This is basically
       *          described by the `RenderProperties` structure which contains all
       *          needed information.
       * @return - the size in bytes needed to describe the properties of the data
       *           needed to perform the computation of such a tile.
       */
      static
      constexpr unsigned
      getPropsSize() noexcept;

      /**
       * @brief - Similar to `getPropsSize` but provides the size in bytes of a
       *          single element of the result buffer of this tile. In the case
       *          of a raytracing tile it is a single `float` which translates
       *          the depth to reach the fractal object.
       * @return - the size in bytes needed to describe a single element of the
       *           result buffer.
       */
      static
      constexpr unsigned
      getResultSize() noexcept;

      /**
       * @brief - Reimplementation of the base class method which basically wraps
       *          the information returned by the `getPropsSize` method. See the
       *          description of this method for more info.
       * @return - the size in bytes for the input parameters of a raytracing job.
       */
      unsigned
      getInputDataSize() override;

      /**
       * @brief - Reimplementation of the base class method to retrieve the dims
       *          of the output buffer needed by this job. In the case of a tile
       *          we just return the size of the internal depth map.
       * @return - the size of the internal depth map.
       */
      utils::Sizei
      getOutputSize() override;

      /**
       * @brief - Reimplementation of the base class method which basically wraps
       *          the information returned by the `getResultSize` method. See the
       *          description of this method for more info.
       * @return - the size in bytes for a single element of the result buffer.
       */
      unsigned
      getOutputDataSize() override;

      /**
       * @brief - Reimplementation of the base class method to provide a pointer
       *          to the actual memory location of the rendering properties.
       * @return - a pointer to the internal rendering properties.
       */
      void*
      getInputData() override;

      /**
       * @brief - Reimplementation of the base class method to expose a pointer to
       *          the internal `m_depthMap` vector where data can be saved.
       * @return - a pointer to the output result buffer for this job.
       */
      void*
      getOutputData() override;

      /**
       * @brief - Retrieves the area associated to this tile. Note that this value
       *          is the same as the one provided when building the object and is
       *          not relevant by itself: the element that created it in the first
       *          place knows how to use it.
       * @return - the area associated to this tile.
       */
      utils::Boxi
      getArea() const noexcept;

      /**
       * @brief - Retrieve the internal data computed by this tile using a depth
       *          map formalism.
       * @return - the depth map computed by this tile.
       */
      const std::vector<float>&
      getDepthMap() const noexcept;

      /**
       * @brief - Returns a distance estimator of the fractal object from the point
       *          `p` in input using the internal properties for determining when
       *          to stop the computation.
       * @param p - the point for which the distance estimator should be computed.
       * @param props - the properties to use to compute the fractal.
       * @return - a floating point value estimating the distance from the point to
       *           the fractal.
       */
      static
      float
      getDistanceEstimator(const utils::Vector3f& p,
                           const RenderProperties& props) noexcept;

    private:

      /**
       * @brief - Used to create and initialize internal variables so that this tile
       *          is ready to be compute. This mainly consists into packaging internal
       *          attributes into a single structure and allocating the internal depth
       *          map which will store the result of the rendering.
       */
      void
      build();

    private:

      /**
       * @brief - The viewpoint of the camera plane. All rays should start
       *          from this point.
       */
      utils::Vector3f m_eye;

      /**
       * @brief - A representation of the `x` axis in the camera plane's
       *          coordinate frame. Allow to distinguihs between horizontal
       *          pixels.
       */
      utils::Vector3f m_u;

      /**
       * @brief - A local representation of the `y` axis. Allows to define
       *          the pixels vertically.
       */
      utils::Vector3f m_v;

      /**
       * @brief - A value describing the local `z` axis of the camera plane.
       *          This allows to effectively move `away` from the plane and
       *          thus get the ray started.
       */
      utils::Vector3f m_w;

      /**
       * @brief - The total area into which this tile is encompassed. Used to
       *          get an idea of the location of the tile in the general cam
       *          plane and be able to correctly generate the rays direction.
       */
      utils::Sizei m_total;

      /**
       * @brief - The part of the camera plane associated to this tile. The
       *          whole area will be computed by this tile and saved as an
       *          internal array which can be fetched by the `Fractal` to
       *          update its own representation afterwards.
       */
      utils::Boxi m_area;

      /**
       * @brief - A general description of the fractal object to compute.
       */
      RenderProperties m_props;

      /**
       * @brief - Translation of all the internal properties into a single
       *          struct that can be passed on to the cuda kernel used to
       *          compute this tile.
       */
      gpu::KernelProps m_cudaProps;

      /**
       * @brief - The depth map accumulated by the process of computing this
       *          tile. The size of this array is determined by the area that
       *          is associated to this tile.
       */
      std::vector<float> m_depthMap;
  };

  using RaytracingTileShPtr = std::shared_ptr<RaytracingTile>;
}

# include "RaytracingTile.hxx"

#endif    /* RAYTRACING_TILE_HH */
