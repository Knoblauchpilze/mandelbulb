#ifndef    RAYTRACING_TILE_HH
# define   RAYTRACING_TILE_HH

# include <mutex>
# include <vector>
# include <memory>
# include <maths_utils/Box.hh>
# include <maths_utils/Vector3.hh>
# include "RenderProperties.hh"
# include "Light.hh"
# include "CudaJob.hh"
# include "RenderProperties.hh"
# include "mandelbulb_kernel.cuh"

namespace mandelbulb {

  namespace pixel {

    /**
     * @brief - Convenience structure representing the data accumulate for a single
     *          pixel during the raytracing. This include the depth to reach the
     *          fractal object from the camera's position (negative value if the
     *          fractal cannot be reached) and the color associated to the pixel.
     */
    struct Data {
      float depth;
      float r;
      float g;
      float b;
    };
  }

  class RaytracingTile: public utils::CudaJob {
    public:

      /**
       * @brief - Create a reaytracing tile with the specified area The area
       *          is usually a small part in a larger total area which dims
       *          are provided through the `total` argument.
       *          Note that the actual properties defining the rendering are
       *          not defined yet and should be populated with the relevant
       *          methods (like `setEye`, etc.). This approach allows to be
       *          able to reuse tiles across the renderings. Indeed most of
       *          the data does not change and as there's quite a lot of mem
       *          allocation going on (and copy from device to host) it makes
       *          sense to try to limit this as much as possible.
       * @param area - the rendering area associated to this area. This is
       *               used to define an area in the camera plane. It will
       *               be used to generate the rays and passed on to the
       *               cuda kernel executing the tile.
       * @param total - the total dimensions of the area this tile is part
       *                of. This helps during the determination of the rays
       *                directions.
       * @param noDataColor - the color to associate to a pixel when it does
       *                      not reach the fractal.
       */
      RaytracingTile(const utils::Boxi& area,
                     const utils::Sizei& total,
                     const sdl::core::engine::Color& noDataColor);

      virtual ~RaytracingTile() = default;

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
       * @brief - Retrieve the internal data computed by this tile using a
       *          pixels map formalism. This contains both the depth to reach
       *          a pixel from the camera's position and the color associated
       *          to it.
       * @return - the pixels map computed by this tile.
       */
      const std::vector<pixel::Data>&
      getPixelsMap() const noexcept;

      void
      setEye(const utils::Vector3f& eye);

      void
      setU(const utils::Vector3f& u);

      void
      setV(const utils::Vector3f& v);

      void
      setW(const utils::Vector3f& w);

      void
      setRenderingProps(const RenderProperties& props);

      void
      setLights(const std::vector<LightShPtr>& lights);

      void
      setNoDataColor(const sdl::core::engine::Color& color);

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
       *          of a raytracing tile it is defined by the `pixel::Data` struct
       *          where all the relevants information is saved.
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
       * @brief - Used to create and initialize internal variables. This mainly is
       *          represented by the pixels map which need to be allocated to be
       *          large enough to hold all the pixels that will be represented by
       *          the tile.
       */
      void
      build();

      /**
       * @brief - Indicate that the `m_dirty` boolean should be set to `true` which
       *          indicates that the `m_cudaProps` attribute should be rebuilt. This
       *          method is typically used whenever one of the internal properties
       *          describing the camera (like `m_u`, etc.) is updated.
       */
      void
      makeDirty();

      /**
       * @brief - Use dinternally whenever the `m_dirty` boolean is set to `true`
       *          which indicates that one of the internal properties has been
       *          changed to update the `m_cudaProps` attribute.
       *          It will reset the `m_dirty` boolean. Note that there's no check
       *          to guarantee that the call is indeed necessary.
       */
      void
      packageCudaProps();

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
       * @brief - The vector of lights defined to illuminate the scene.
       *          Will be converted into a flattened representation when
       *          passed to the cuda kernels.
       */
      std::vector<LightShPtr> m_lights;

      /**
       * @brief - The color to assign to pixels that do not reach the fractal.
       */
      sdl::core::engine::Color m_noDataColor;

      /**
       * @brief - Used to determine whether the internal properties have been
       *          changed since the last called to `packageCudaProps`. This is
       *          important to guarantee the synchronization between the props
       *          and their cuda equivalent and to guarantee that we can reuse
       *          the tile across renderings.
       */
      bool m_dirty;

      /**
       * @brief - Translation of all the internal properties into a single
       *          struct that can be passed on to the cuda kernel used to
       *          compute this tile.
       */
      gpu::KernelProps m_cudaProps;

      /**
       * @brief - The pixels map accumulated by the process of computing this
       *          tile. This contains both the color associated to each pixel
       *          but also the depth to travel from the camera's position to
       *          reach the fractal object.
       *          In case the fractal cannot be reached a negative value is
       *          used for the depth. The size of this map is always equal to
       *          the area of the `m_area` attribute.
       */
      std::vector<pixel::Data> m_pixelsMap;
  };

  using RaytracingTileShPtr = std::shared_ptr<RaytracingTile>;
}

# include "RaytracingTile.hxx"

#endif    /* RAYTRACING_TILE_HH */
