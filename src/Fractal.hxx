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

    updateFromCamera();
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

    updateFromCamera();
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
    return RaytracingTile::getDistanceEstimator(m_camera->getEye(), m_props);
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

    updateFromCamera();
  }

  inline
  utils::Sizei
  Fractal::getData(std::vector<float>& depths) {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Copy the internal data to the output vector after resizing it
    // if needed.
    if (depths.size() != m_samples.size()) {
      depths.resize(m_samples.size());
    }

    // Fill the depths values.
    for (unsigned id = 0u ; id < m_samples.size() ; ++id) {
      depths[id] = m_samples[id].depth;
    }

    // Return the dimensions of the fractal.
    return m_dims;
  }

  inline
  float
  Fractal::getPoint(const utils::Vector2i& screenCoord,
                    utils::Vector3f& worldCoord,
                    bool& hit)
  {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Assume no hit.
    hit = false;

    worldCoord.x() = std::numeric_limits<float>::lowest();
    worldCoord.y() = std::numeric_limits<float>::lowest();
    worldCoord.z() = std::numeric_limits<float>::lowest();

    float depth = -1.0f;

    // Convert to local coordinates.
    utils::Vector2f lScreen(
      screenCoord.x() + m_dims.w() / 2,
      screenCoord.y() + m_dims.h() / 2
    );

    // Consistency check
    if (lScreen.x() < 0 || lScreen.x() >= m_dims.w() ||
        lScreen.y() < 0 || lScreen.y() >= m_dims.h())
    {
      log(
        std::string("Trying to get point at coord ") + lScreen.toString() +
        " not compatible with internal camera plane size " + m_dims.toString(),
        utils::Level::Error
      );

      return -1.0f;
    }

    // Retrieve the depth at this point: this will be used both to fill
    // the return value and to get the real world coordinates of the
    // point located at said screen coordinates.
    int off = lScreen.y() * m_dims.w() + lScreen.x();
    depth = m_samples[off].depth / m_samples[off].iter;

    // Check whether we have a hit.
    if (depth < 0.0f) {
      return depth;
    }

    // We have a hit !
    hit = true;

    // Use the camera to update the real world coordinate.
    utils::Vector2f perc(
      -0.5f + 1.0f * lScreen.x() / m_dims.w(),
      -0.5f + 1.0f * lScreen.y() / m_dims.h()
    );

    utils::Vector3f dir = m_camera->getDirection(perc);
    worldCoord = m_camera->getEye() + depth * dir;

    log("Screen: " + screenCoord.toString() + " dir: " + dir.toString() + ", depth: " + std::to_string(depth), utils::Level::Verbose);

    return depth;
  }

  inline
  void
  Fractal::onRenderingPropsChanged(RenderProperties props) {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Update the internal properties.
    m_props = props;

    // Reset existing results.
    std::fill(m_samples.begin(), m_samples.end(), Sample{0u, -1.0f});

    // Perform a new rendering.
    scheduleRendering(true);
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
  Fractal::updateFromCamera() {
    // Reset existing results.
    std::fill(m_samples.begin(), m_samples.end(), Sample{0u, -1.0f});

    // Set the results to be accumulating and schedule a rendering.
    // We want a complete recompute of the iterations so we need to
    // reset everything regarding the progression.
    m_computationState = State::Accumulating;
    scheduleRendering(true);
  }

}

#endif    /* FRACTAL_HXX */
