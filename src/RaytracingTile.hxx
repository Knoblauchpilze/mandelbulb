#ifndef    RAYTRACING_TILE_HXX
# define   RAYTRACING_TILE_HXX

# include "RaytracingTile.hh"

namespace mandelbulb {

  inline
  constexpr unsigned
  RaytracingTile::getPropsSize() noexcept {
    return sizeof(RenderProperties);
  }

  inline
  constexpr unsigned
  RaytracingTile::getResultSize() noexcept {
    return sizeof(float);
  }

  inline
  unsigned
  RaytracingTile::getInputDataSize() {
    return getPropsSize();
  }

  inline
  unsigned
  RaytracingTile::getOutputDataSize() {
    return getResultSize();
  }

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

}

#endif    /* RAYTRACING_TILE_HXX */
