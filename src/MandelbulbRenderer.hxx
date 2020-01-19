#ifndef    MANDELBULB_RENDERER_HXX
# define   MANDELBULB_RENDERER_HXX

# include "MandelbulbRenderer.hh"

namespace mandelbulb {

  inline
  void
  MandelbulbRenderer::updatePrivate(const utils::Boxf& window) {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Use the base handler.
    sdl::core::SdlWidget::updatePrivate(window);

    // Update the rendering options if needed.
    if (m_fractal != nullptr) {
      m_fractal->setCameraDims(window.toSize());
    }
  }

  inline
  void
  MandelbulbRenderer::onTilesRendered() {
    // TODO: Implementation, use a similar mechanism as for the `ColonyRenderer`
    // with textures that can be invalidated.
  }

}

#endif    /* MANDELBULB_RENDERER_HXX */
