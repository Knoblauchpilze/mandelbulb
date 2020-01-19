
# include "MandelbulbRenderer.hh"

namespace mandelbulb {

  MandelbulbRenderer::MandelbulbRenderer(FractalShPtr fractal,
                                         const utils::Sizef& hint,
                                         sdl::core::SdlWidget* parent):
    sdl::graphic::ScrollableWidget(std::string("renderer"),
                                   parent,
                                   hint),

    m_propsLocker(),
    m_fractal(fractal),

    onCoordinatesChanged(),
    onDepthChanged()
  {
    setService(std::string("mandelbulb"));

    // Consistency check.
    if (m_fractal == nullptr) {
      error(
        std::string("Could not create mandelbulb renderer"),
        std::string("Invalid null fractal object to display")
      );
    }

    build();
  }

  void
  MandelbulbRenderer::build() {
    // Connect the fractal signal indicating that some tiles have been rendered to the
    // local slot allowing to schedule a repaint.
    m_fractal->onTilesRendered.connect_member<MandelbulbRenderer>(
      this,
      &MandelbulbRenderer::onTilesRendered
    );
  }

}
