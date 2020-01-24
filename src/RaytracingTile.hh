#ifndef    RAYTRACING_TILE_HH
# define   RAYTRACING_TILE_HH

# include <mutex>
# include <vector>
# include <memory>
# include <core_utils/AsynchronousJob.hh>
# include <maths_utils/Box.hh>
# include <maths_utils/Vector3.hh>
# include "RenderProperties.hh"

namespace mandelbulb {

  class RaytracingTile: public utils::AsynchronousJob {
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

      ~RaytracingTile() = default;

      /**
       * @brief - Reimplementation of the interface method allowing to perform
       *          the computation of the fractal's data for the area associated
       *          to this tile.
       */
      void
      compute() override;

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
       * @brief - Define a suitable default value for subpixel jittering. This will
       *          be applied to offset slightly the ray direction when generating it
       *          so that we get different rays for each pixel and thus obtain some
       *          sort of built-in antialiasing.
       * @return - a suited value for subpixel jittering.
       */
      static
      constexpr float
      getJitteringRadius() noexcept;

      /**
       * @brief - Used to generate a ray direction from the pixel at coordinates
       *          `(x, y)`. This will use the internal vectors `m_u/v/w` and some
       *          jittering information so that we get some built-in antialiasing.
       * @param x - the `x` coordinate of the pixel for which the ray direction is
       *            to be generated.
       * @param y - the `y` coordinate of the pixel for which the ray direction is
       *            to be generated.
       * @return - the direction for the corresponding pixel.
       */
      utils::Vector3f
      generateRayDir(int x,
                     int y) const;

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
