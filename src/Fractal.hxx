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

    // Update the internal camera.
    // TODO: Note that we should probably change the constructor of the
    // `RaytracingTile` so as not to take a pointer otherwise in case we
    // change the camera when some jobs are running we might use the new
    // camera (which might be okay but better not risk it).
    // TODO: We should handle the cast to integer dimensions and reflect
    // it in the internal data array.
    bool changed = m_camera->setDims(
      utils::Sizei(
        static_cast<int>(std::floor(dims.w())),
        static_cast<int>(std::floor(dims.h()))
      )
    );

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

}

#endif    /* FRACTAL_HXX */
