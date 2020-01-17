
# include "MandelbulbRenderer.hh"

namespace mandelbulb {

  MandelbulbRenderer::MandelbulbRenderer(const utils::Sizef& hint,
                                         sdl::core::SdlWidget* parent):
    sdl::graphic::ScrollableWidget(std::string("renderer"),
                                   parent,
                                   hint),

    m_propsLocker()
  {
    setService(std::string("mandelbulb"));

    build();
  }

  void
  MandelbulbRenderer::build() {}

}
