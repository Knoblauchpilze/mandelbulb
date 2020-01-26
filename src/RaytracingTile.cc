
# include "RaytracingTile.hh"

namespace mandelbulb {

  RaytracingTile::RaytracingTile(const utils::Vector3f& eye,
                                 const utils::Vector3f& u,
                                 const utils::Vector3f& v,
                                 const utils::Vector3f& w,
                                 const utils::Sizei& total,
                                 const utils::Boxi& area,
                                 const RenderProperties& props):
    utils::CudaJob(std::string("tile_") + area.toString()),

    m_eye(eye),

    m_u(u),
    m_v(v),
    m_w(w),

    m_total(total),
    m_area(area),

    m_props(props),

    m_cudaProps(),

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

    build();
  }

  float
  RaytracingTile::getDistanceEstimator(const utils::Vector3f& p,
                                       const RenderProperties& props) noexcept {
    // Compute as many iterations as needed.
    unsigned iter = 0u;
    utils::Vector3f z = p;
    float r = z.length();

    float theta, phi;
    float dr = 1.0f;

    while (iter < props.accuracy && r < props.bailout) {
      // Detect escaping series.
      r = z.length();

      if (r < props.bailout) {
        // Convert to polar coordinates.
        theta = std::acos(z.z() / r);
        phi = std::atan2(z.y(), z.x());

        // Update distance estimator.
        dr = std::pow(r, props.exponent - 1.0f) * props.exponent * dr + 1.0f;

        // Scale and rotate the point.
        float zr = std::pow(r, props.exponent);
        theta *= props.exponent;
        phi *= props.exponent;

        // Convert back to cartesian coordinates.
        z.x() = zr * std::cos(phi) * std::sin(theta);
        z.y() = zr * std::sin(phi) * std::sin(theta);
        z.z() = zr * std::cos(theta);

        z += p;
      }

      ++iter;
    }

    // Return the distance estimator.
    return 0.5f * std::log(r) * r / dr;
  }


  void
  RaytracingTile::build() {
    // Allocate the internal vector array.
    m_depthMap.resize(m_area.w() * m_area.h() * 4u, 0.0f);
    for (int id = 0 ; id < m_area.area() ; ++id) {
      m_depthMap[4u * id + 3u] = -1.0f;
    }

    // Package the internal properties into a valid `KernelProps` struct.
    m_cudaProps = gpu::KernelProps{
      m_props.accuracy,
      m_props.exponent,
      m_props.bailout,
      m_props.hitThreshold,
      m_props.raySteps,

      m_eye.x(),
      m_eye.y(),
      m_eye.z(),

      m_u.x(),
      m_u.y(),
      m_u.z(),

      m_v.x(),
      m_v.y(),
      m_v.z(),

      m_w.x(),
      m_w.y(),
      m_w.z(),

      m_area.getLeftBound(),
      m_area.getBottomBound(),
      m_total.w(),
      m_total.h()
    };
  }

}
