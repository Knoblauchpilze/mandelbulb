#ifndef    RAYTRACING_TILE_HH
# define   RAYTRACING_TILE_HH

# include <mutex>
# include <memory>
# include <core_utils/AsynchronousJob.hh>
# include <maths_utils/Box.hh>
# include <maths_utils/Vector3.hh>

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
       * @param area - the area of the camera plane which should be
       *               processed by this tile.
       */
      RaytracingTile(const utils::Vector3f& eye,
                     const utils::Vector3f& u,
                     const utils::Vector3f& v,
                     const utils::Vector3f& w,
                     const utils::Boxi& area);

      ~RaytracingTile() = default;

      /**
       * @brief - Reimplementation of the interface method allowing to perform
       *          the computation of the fractal's data for the area associated
       *          to this tile.
       */
      void
      compute() override;

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
       * @brief - The part of the camera plane associated to this tile. The
       *          whole area will be computed by this tile and saved as an
       *          internal array which can be fetched by the `Fractal` to
       *          update its own representation afterwards.
       */
      utils::Boxi m_area;
  };

  using RaytracingTileShPtr = std::shared_ptr<RaytracingTile>;
}

# include "RaytracingTile.hxx"

#endif    /* RAYTRACING_TILE_HH */
