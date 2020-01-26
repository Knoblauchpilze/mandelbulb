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
  const std::vector<pixel::Data>&
  RaytracingTile::getPixelsMap() const noexcept {
    return m_pixelsMap;
  }

  inline
  void
  RaytracingTile::setEye(const utils::Vector3f& eye) {
    m_eye = eye;
    makeDirty();
  }

  inline
  void
  RaytracingTile::setU(const utils::Vector3f& u) {
    m_u = u;
    makeDirty();
  }

  inline
  void
  RaytracingTile::setV(const utils::Vector3f& v) {
    m_v = v;
    makeDirty();
  }

  inline
  void
  RaytracingTile::setW(const utils::Vector3f& w)  {
    m_w = w;
    makeDirty();
  }

  inline
  void
  RaytracingTile::setRenderingProps(const RenderProperties& props) {
    m_props = props;
    makeDirty();
  }

  inline
  void
  RaytracingTile::setLights(const std::vector<LightShPtr>& lights) {
    m_lights = lights;
    makeDirty();
  }

  inline
  void
  RaytracingTile::setNoDataColor(const sdl::core::engine::Color& color) {
    m_noDataColor = color;
    makeDirty();
  }

  inline
  constexpr unsigned
  RaytracingTile::getPropsSize() noexcept {
    return sizeof(gpu::KernelProps);
  }

  inline
  constexpr unsigned
  RaytracingTile::getResultSize() noexcept {
    return sizeof(pixel::Data);
  }

  inline
  unsigned
  RaytracingTile::getInputDataSize() {
    return getPropsSize();
  }

  inline
  utils::Sizei
  RaytracingTile::getOutputSize() {
    return m_area.toSize();
  }

  inline
  unsigned
  RaytracingTile::getOutputDataSize() {
    return getResultSize();
  }

  inline
  void*
  RaytracingTile::getInputData() {
    // Verify whether we need to package the props.
    if (m_dirty) {
      packageCudaProps();
    }

    return &m_cudaProps;
  }

  inline
  void*
  RaytracingTile::getOutputData() {
    return reinterpret_cast<void*>(m_pixelsMap.data());
  }

  inline
  void
  RaytracingTile::makeDirty() {
    m_dirty = true;
  }

}

#endif    /* RAYTRACING_TILE_HXX */
