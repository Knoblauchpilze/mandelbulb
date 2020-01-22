
# include "RaytracingTile.hh"

namespace mandelbulb {

  RaytracingTile::RaytracingTile(const utils::Vector3f& eye,
                                 const utils::Vector3f& u,
                                 const utils::Vector3f& v,
                                 const utils::Vector3f& w,
                                 const utils::Sizei& total,
                                 const utils::Boxi& area):
    utils::AsynchronousJob(std::string("tile_") + area.toString()),

    m_eye(eye),

    m_u(u),
    m_v(v),
    m_w(w),

    m_total(total),
    m_area(area),

    m_depthMap()
  {
    // Check consistency.
    if (!m_total.valid()) {
      error(
        std::string("Could not create rendering tile for area ") + m_area.toString(),
        std::string("Invalid total area ") + m_total.toString()
      );
    }
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
        // For each pixel of the tile we need to generate a valid ray direction
        // to perform the computation. We don't have the camera but we can use
        // the internal vectors.
        utils::Vector3f dir = generateRayDir(x, y);

        m_depthMap[y * m_area.w() + x] = std::abs(dir.z());
      }
    }

    // TODO: Implementation. Note that the internal area should not include the
    // right and top bounds ?
  }

  utils::Vector3f
  RaytracingTile::generateRayDir(int x,
                                 int y) const
  {
    // Compute the general direction of the ray from the input
    // coordinates.
    int lX = x + m_area.getLeftBound() + m_total.w() / 2;
    int lY = y + m_area.getBottomBound() + m_total.h() / 2;

    // Handle some jittering.
    float rnd = 1.0f * std::rand() / RAND_MAX;
    float jX = 1.0f * lX + std::cos(rnd) * getJitteringRadius();
    float jY = 1.0f * lY + std::sin(rnd) * getJitteringRadius();

    // Express this pixel's value into a percentage of the total
    // area and offset it to obtain a centered value.
    float percX = -0.5f + jX / m_total.w();
    float percY = -0.5f + jY / m_total.h();

    // This can be used to compute the ray direction.
    utils::Vector3f rawDir = percX * m_u + percY * m_v + m_w;

    return rawDir.normalized();
  }

}
