#ifndef    MANDELBULB_RENDERER_HXX
# define   MANDELBULB_RENDERER_HXX

# include "MandelbulbRenderer.hh"

namespace mandelbulb {

  inline
  MandelbulbRenderer::~MandelbulbRenderer() {
    Guard guard(m_propsLocker);
  }

}

#endif    /* MANDELBULB_RENDERER_HXX */
