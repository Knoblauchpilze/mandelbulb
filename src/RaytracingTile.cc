
# include "RaytracingTile.hh"

namespace mandelbulb {

  RaytracingTile::RaytracingTile(const utils::Boxi& area,
                                 const utils::Sizei& total):
    utils::CudaJob(std::string("tile_") + area.toString()),

    m_eye(),

    m_u(),
    m_v(),
    m_w(),

    m_total(total),
    m_area(area),

    m_rProps(),
    m_sProps(),
    m_lights(),

    m_dirty(true),
    m_cudaProps(),

    m_pixelsMap()
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

    // Return the distance estimator. See the `mandelbulb_kernel.cu` file for
    // a more accurate discussion about this formula.
    return 0.25f * std::log(r) * r / dr;
  }

  void
  RaytracingTile::build() {
    // Allocate the internal pixels map.
    m_pixelsMap.resize(
      m_area.w() * m_area.h(),
      pixel::Data{
        -1.0f,
        m_sProps.noDataColor.r(),
        m_sProps.noDataColor.g(),
        m_sProps.noDataColor.b()
      }
    );
  }

  void
  RaytracingTile::packageCudaProps() {
    // Package the internal properties into their cuda equivalent structure.
    // We will flatten the lights and register the missing one as inactive
    // if needed.
    m_cudaProps.accuracy = m_rProps.accuracy;
    m_cudaProps.exponent =  m_rProps.exponent;
    m_cudaProps.bailout =  m_rProps.bailout;
    m_cudaProps.hit_thresh =  m_rProps.hitThreshold;
    m_cudaProps.ray_steps =  m_rProps.raySteps;

    m_cudaProps.eye_x = m_eye.x();
    m_cudaProps.eye_y = m_eye.y();
    m_cudaProps.eye_z = m_eye.z();

    m_cudaProps.u_x = m_u.x();
    m_cudaProps.u_y = m_u.y();
    m_cudaProps.u_z = m_u.z();

    m_cudaProps.v_x = m_v.x();
    m_cudaProps.v_y = m_v.y();
    m_cudaProps.v_z = m_v.z();

    m_cudaProps.w_x = m_w.x();
    m_cudaProps.w_y = m_w.y();
    m_cudaProps.w_z = m_w.z();

    m_cudaProps.min_x = m_area.getLeftBound();
    m_cudaProps.min_y = m_area.getBottomBound();
    m_cudaProps.tot_w = m_total.w();
    m_cudaProps.tot_h = m_total.h();

    m_cudaProps.f_r = m_sProps.fColor.r();
    m_cudaProps.f_g = m_sProps.fColor.g();
    m_cudaProps.f_b = m_sProps.fColor.b();

    m_cudaProps.no_data_r = m_sProps.noDataColor.r();
    m_cudaProps.no_data_g = m_sProps.noDataColor.g();
    m_cudaProps.no_data_b = m_sProps.noDataColor.b();

    m_cudaProps.blending = m_sProps.blending;

    // Copy lights.
    for (unsigned id = 0u ; id < m_lights.size() && id < MAX_LIGHTS ; ++id) {
      // Check consistency.
      if (m_lights[id] == nullptr) {
        log(
          std::string("Could not copy invalid null light ") + std::to_string(id) + " to cuda kernel props",
          utils::Level::Error
        );

        continue;
      }

      Light& l = *m_lights[id];

      LIGHT_PROP(m_cudaProps.lights, id, gpu::ACTIVE) = 1.0f;

      utils::Vector3f dir = l.getDirection();
      LIGHT_PROP(m_cudaProps.lights, id, gpu::DIR_X) = dir.x();
      LIGHT_PROP(m_cudaProps.lights, id, gpu::DIR_Y) = dir.y();
      LIGHT_PROP(m_cudaProps.lights, id, gpu::DIR_Z) = dir.z();

      sdl::core::engine::Color c = l.getColor();
      LIGHT_PROP(m_cudaProps.lights, id, gpu::COLOR_R) = c.r();
      LIGHT_PROP(m_cudaProps.lights, id, gpu::COLOR_G) = c.g();
      LIGHT_PROP(m_cudaProps.lights, id, gpu::COLOR_B) = c.b();

      LIGHT_PROP(m_cudaProps.lights, id, gpu::INTENSITY) = l.getIntensity();
    }

    // We also need to deactivate the lights that are not defined
    // by the internal vector.
    if (m_lights.size() < MAX_LIGHTS) {
      for (unsigned id = m_lights.size() ; id < MAX_LIGHTS ; ++id) {
        LIGHT_PROP(m_cudaProps.lights, id, gpu::ACTIVE) = -1.0f;
      }
    }

    // Tonemap properties.
    m_cudaProps.exposure = m_sProps.exposure;
    m_cudaProps.burnout = m_sProps.burnout;
  }

}
