
# include "RaytracingTile.hh"

namespace mandelbulb {

  RaytracingTile::RaytracingTile(const utils::Vector3f& eye,
                                 const utils::Vector3f& u,
                                 const utils::Vector3f& v,
                                 const utils::Vector3f& w,
                                 const utils::Boxi& area):
    utils::AsynchronousJob(std::string("tile_") + area.toString()),

    m_eye(eye),

    m_u(u),
    m_v(v),
    m_w(w),

    m_area(area)
  {
    // Check consistency.
    if (!m_area.valid()) {
      error(
        std::string("Could not create rendering tile for area ") + m_area.toString(),
        std::string("Invalid area")
      );
    }
  }

  void
  RaytracingTile::compute() {
    // TODO: Implementation. Note that the internal area should not include the
    // right and top bounds ?
    log("Should compute data for area " + m_area.toString(), utils::Level::Warning);
  }

}
