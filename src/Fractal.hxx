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
      // TODO: Potential deadlock ?
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

}

#endif    /* FRACTAL_HXX */
