
# include "FractalSettings.hh"

namespace mandelbulb {

  FractalSettings::FractalSettings(const utils::Sizef& sizeHint,
                                   sdl::core::SdlWidget* parent):
    sdl::core::SdlWidget(std::string("fractal_settings"),
                         sizeHint,
                         parent,
                         getBackgroundColor()),

    m_propsLocker()
  {
    build();
  }

  void
  FractalSettings::build() {
    // TODO: Implementation.
  }

  void
  FractalSettings::onApplyButtonClicked(const std::string& /*dummy*/) {
    // TODO: Implementation.
  }

}
