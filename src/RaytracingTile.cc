
# include "RaytracingTile.hh"

namespace mandelbulb {

  RaytracingTile::RaytracingTile(const utils::Vector3f& eye,
                                 const utils::Vector3f& u,
                                 const utils::Vector3f& v,
                                 const utils::Vector3f& w,
                                 const utils::Sizei& total,
                                 const utils::Boxi& area,
                                 const RenderProperties& props):
    utils::AsynchronousJob(std::string("tile_") + area.toString()),

    m_eye(eye),

    m_u(u),
    m_v(v),
    m_w(w),

    m_total(total),
    m_area(area),

    m_props(props),

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
        // Generate a direction for the ray to march on.
        utils::Vector3f dir = generateRayDir(x, y);

        // March on this ray until we reach a point close
        // enough from the fractal.
        float tDist = 0.0f;
        unsigned steps = 0u;
        float dist = getSurfaceHitThreshold() + 1.0f;

        utils::Vector3f p = m_eye + tDist * dir;

        while (steps < getMaxRaySteps() && dist > getSurfaceHitThreshold()) {
          // March on the ray.
          p = m_eye + tDist * dir;

          // Get an estimation of the distance.
          dist = getDistanceEstimator(p);

          // Add this and move on to the next step.
          tDist += dist;
          ++steps;
        }

        if (dist <= getSurfaceHitThreshold()) {
          m_depthMap[y * m_area.w() + x] = tDist;
        }
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

  float
  RaytracingTile::getDistanceEstimator(const utils::Vector3f& p) const noexcept {
    // Compute as many iterations as needed.
    unsigned iter = 0u;
    utils::Vector3f z = p;
    float r = z.length();

    float theta, phi;
    float dr = 1.0f;

    while (iter < m_props.iterations && r < getBailoutDistance()) {
      // Detect escaping series.
      r = z.length();

      if (r < getBailoutDistance()) {
        // Convert to polar coordinates.
        theta = std::acos(z.z() / r);
        phi = std::atan2(z.y(), z.x());

        // Update distance estimator.
        dr = std::pow(r, m_props.exponent - 1.0f) * m_props.exponent * dr + 1.0f;

        // Scale and rotate the point.
        float zr = std::pow(r, m_props.exponent);
        theta *= m_props.exponent;
        phi *= m_props.exponent;

        // Convert back to cartesian coordinates.
        z.x() = zr * std::cos(phi) * std::sin(theta);
        z.y() = zr * std::sin(phi) * std::sin(theta);
        z.z() = zr * std::cos(theta);

        z += p;
      }

      ++iter;
    }
    // TODO: Right now the computation stalls directly at the starting point of the
    // ray because we're at `(0,-5,0)` so it is greater than the bailout distance
    // so we end up with a distance estimator of `0.5*ln(5)*5/1 ~ 4` we match of `4`
    // on the ray and end up almost in the fractal (I guess) so every point lies in
    // it.
    // Maybe we should check that the direction are actually correct because it is
    // really strange that seeing the fractal from `5` units away we still end up
    // with it occupying all the screen.

    // Return the distance estimator.
    return 0.5f * std::log(r) * r / dr;
  }

}
