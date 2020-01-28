#ifndef    FRACTAL_HXX
# define   FRACTAL_HXX

# include "Fractal.hh"

namespace mandelbulb {

  inline
  Fractal::~Fractal() {
    // Stop the scheduler so that we don't continue processing
    // tiles while it's obviously not needed anymore.
    Guard guard(m_propsLocker);

    m_scheduler->onJobsCompleted.disconnect(m_tilesRenderedSignalID);
    m_scheduler.reset();
  }

  inline
  void
  Fractal::setCameraDims(const utils::Sizef& dims) {
    // Check consistency.
    if (!dims.valid()) {
      error(
        std::string("Could not update camera for fractal"),
        std::string("Invalid null dimensions ") + dims.toString()
      );
    }

    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Update the internal camera: to do so we need to convert the
    // input area into integer dimensions. In order to be sure to
    // cover at least the input dimensions we will expand the input
    // size a bit.
    utils::Sizei iDims(
      static_cast<int>(std::ceil(dims.w())),
      static_cast<int>(std::ceil(dims.h()))
    );

    // Make sure that the dimensions are always even: this will make
    // things a lot easier when creating the tiles. And anyways we
    // can still account for this when we are requested the data at
    // specific coordinates as long as we know what we're representing
    // internally.
    iDims.w() += iDims.w() % 2;
    iDims.h() += iDims.h() % 2;

    bool changed = m_camera->setDims(iDims);

    // In case the dimensions where actually changed we need to update
    // the internal attributes and schedule a new rendering.
    if (!changed) {
      return;
    }

    m_dims = iDims;
    m_samples.resize(m_dims.area());

    updateAndRender();
  }

  inline
  void
  Fractal::onLightsChanged(const std::vector<LightShPtr>& lights) {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Update the internal lights vector.
    m_lights = lights;

    // Update and render the fractal with the new lights.
    updateAndRender();
  }

  inline
  void
  Fractal::rotateCamera(const utils::Vector2f& rotations) {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Update the camera.
    bool changed = m_camera->rotate(rotations);

    // In case the rotations did not modify the current state of the
    // camera we don't need to do anything more. Otherwise we should
    // schedule a repaint and notify external listeners.
    if (!changed) {
      return;
    }

    updateAndRender();
  }

  inline
  float
  Fractal::getDistance() noexcept {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    return m_camera->getDistance();
  }

  inline
  float
  Fractal::getDistanceEstimation() noexcept {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Retrieve the current eye's position and compute
    // an estimation of the distance to the fractal for
    // this position.
    return RaytracingTile::getDistanceEstimator(m_camera->getEye(), m_rProps);
  }

  inline
  void
  Fractal::updateDistance(float dist) {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Update the camera.
    bool changed = m_camera->setDistance(dist);

    // In case the rotations did not modify the current state of the
    // camera we don't need to do anything more. Otherwise we should
    // schedule a repaint and notify external listeners.
    if (!changed) {
      return;
    }

    updateAndRender();
  }

  inline
  utils::Sizei
  Fractal::getData(std::vector<sdl::core::engine::Color>& colors) {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Copy the internal data to the output vector after resizing it
    // if needed.
    if (colors.size() != m_samples.size()) {
      colors.resize(m_samples.size());
    }

    // Fill the colors values.
    for (unsigned id = 0u ; id < m_samples.size() ; ++id) {
      colors[id] = m_samples[id].color;
    }

    // Return the dimensions of the fractal.
    return m_dims;
  }

  inline
  void
  Fractal::onRenderingPropsChanged(RenderProperties props) {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Update the internal properties.
    m_rProps = props;

    // Schedule rendering.
    updateAndRender();
  }

  inline
  void
  Fractal::onShadingPropsChanged(ShadingProperties props) {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Update the internal properties.
    m_sProps = props;

    // Schedule rendering.
    updateAndRender();
  }

  inline
  unsigned
  Fractal::getWorkerThreadCount() noexcept {
    return 3u;
  }

  inline
  constexpr unsigned
  Fractal::getTileWidth() noexcept {
    return 150u;
  }

  inline
  constexpr unsigned
  Fractal::getTileHeight() noexcept {
    return 100;
  }

  inline
  void
  Fractal::updateAndRender() {
    // Reset existing results.
    std::fill(
      m_samples.begin(),
      m_samples.end(),
      Sample{-1.0f, m_sProps.noDataColor}
    );

    // TODO: We should probably update the schedule at this point.

    // Set the results to be accumulating and schedule a rendering.
    // We want a complete recompute of the iterations so we need to
    // reset everything regarding the progression.
    m_computationState = State::Accumulating;
    scheduleRendering();
  }

}

#endif    /* FRACTAL_HXX */
