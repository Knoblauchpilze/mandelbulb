#ifndef    FRACTAL_HXX
# define   FRACTAL_HXX

# include "Fractal.hh"

namespace mandelbulb {

  inline
  Fractal::~Fractal() {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Stop the scheduler so that we don't continue processing
    // tiles while it's obviously not needed anymore.
    if (m_scheduler != nullptr) {
      m_scheduler.reset();
    }
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
    std::fill(m_samples.begin(), m_samples.end(), Sample{1u, -1.0f});

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
  void
  Fractal::updateDistance(float factor) {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Update the camera.
    bool changed = m_camera->setDistance(factor);

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
  Fractal::getPoint(const utils::Vector2i& screenCoord,
                    utils::Vector3f& worldCoord,
                    bool& hit)
  {
    // TODO: Implementation.
    log("Should get point under " + screenCoord.toString());

    hit = false;

    worldCoord.x() = std::numeric_limits<float>::lowest();
    worldCoord.y() = std::numeric_limits<float>::lowest();
    worldCoord.z() = std::numeric_limits<float>::lowest();

    return -1.0f;
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
    // Set the results to be accumulating and schedule a rendering.
    // We want a complete recompute of the iterations so we need to
    // reset everything regarding the progression.
    m_computationState = State::Accumulating;
    scheduleRendering(true);

    // Notify listeners that a new camera exists.
    onCameraChanged.safeEmit(
      std::string("onCameraChanged(") + m_camera->getEye().toString() + ")",
      m_camera
    );
  }

}

#endif    /* FRACTAL_HXX */