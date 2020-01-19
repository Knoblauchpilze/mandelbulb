#ifndef    RENDERER_SCHEDULER_HXX
# define   RENDERER_SCHEDULER_HXX

# include "RendererScheduler.hh"

namespace mandelbulb {

  inline
  RendererScheduler::~RendererScheduler() {
    // Stops the rendering.
    stop();
  }

  inline
  unsigned
  RendererScheduler::getWorkerThreadCount() noexcept {
    return 3u;
  }

}

#endif    /* RENDERER_SCHEDULER_HXX */
