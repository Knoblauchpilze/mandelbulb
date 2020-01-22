#ifndef    RAYTRACING_TILE_HXX
# define   RAYTRACING_TILE_HXX

# include "RaytracingTile.hh"

namespace mandelbulb {

  inline
  utils::Boxi
  RaytracingTile::getArea() const noexcept {
    return m_area;
  }

  inline
  const std::vector<float>&
  RaytracingTile::getDepthMap() const noexcept {
    return m_depthMap;
  }

  inline
  constexpr float
  RaytracingTile::getJitteringRadius() noexcept {
    return 0.1f;
  }

  inline
  constexpr float
  RaytracingTile::getBailoutDistance() noexcept {
    return 4.0f;
  }

  inline
  constexpr float
  RaytracingTile::getSurfaceHitThreshold() noexcept {
    return 0.01f;
  }

  inline
  constexpr unsigned
  RaytracingTile::getMaxRaySteps() noexcept {
    return 10u;
  }

}

#endif    /* RAYTRACING_TILE_HXX */
