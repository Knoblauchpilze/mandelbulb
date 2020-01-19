
# include "RaytracingTile.hh"

namespace mandelbulb {

  RaytracingTile::RaytracingTile(CameraShPtr camera,
                                 const utils::Boxi& area):
    utils::AsynchronousJob(std::string("tile_") + area.toString()),

    m_camera(camera),
    m_area(area)
  {
    // Check consistency.
    if (m_camera == nullptr) {
      error(
        std::string("Could not create rendering tile for area ") + m_area.toString(),
        std::string("Invalid null camera")
      );
    }
    if (!m_area.valid()) {
      error(
        std::string("Could not create rendering tile for area ") + m_area.toString(),
        std::string("Invalid area")
      );
    }
  }

}
