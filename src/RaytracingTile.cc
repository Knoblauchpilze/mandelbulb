
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

    m_area(area),

    m_depthMap()
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
    // Allocate the internal vector array.
    m_depthMap.resize(m_area.w() * m_area.h(), -1.0f);

    for (int y = 0 ; y < m_area.h() ; ++y) {
      for (int x = 0 ; x < m_area.w() ; ++x) {
        float fx = (x - m_area.w() / 2.0f) / m_area.w();
        float fy = (y - m_area.h() / 2.0f) / m_area.h();

        m_depthMap[y * m_area.w() + x] = 10.0f * std::exp(-(fx * fx + fy * fy));
      }
    }

    // TODO: Implementation. Note that the internal area should not include the
    // right and top bounds ?
    // log("Should compute data for area " + m_area.toString(), utils::Level::Warning);
  }

}
