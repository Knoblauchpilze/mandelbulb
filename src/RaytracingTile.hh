#ifndef    RAYTRACING_TILE_HH
# define   RAYTRACING_TILE_HH

# include <mutex>
# include <memory>
# include <core_utils/AsynchronousJob.hh>
# include <maths_utils/Box.hh>
# include "Camera.hh"

namespace mandelbulb {

  class RaytracingTile: public utils::AsynchronousJob {
    public:

      /**
       * @brief - Associates this raytracing tile with the specified
       *          camera and an area which should represent a part of
       *          the camera plane.
       *          An error is raised if either the `camera` or the
       *          `area` are invalid.
       * @param camera - the camera allowing to get a viewpoint on the
       *                 fractal object.
       * @param area - the area of the camera plane which should be
       *               processed by this tile.
       */
      RaytracingTile(CameraShPtr camera,
                     const utils::Boxi& area);

      ~RaytracingTile() = default;

    private:

      /**
       * @brief - Defines the viewpoint on the fractal to compute through
       *          this tile. The camera plane is divided into several parts
       *          each one being assigned a dedicated raytracing tile. It
       *          allows to schedule the computation in parallel as each
       *          tile has a distinct area associated to it representing a
       *          distinct portion of the camera plane.
       */
      CameraShPtr m_camera;

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
