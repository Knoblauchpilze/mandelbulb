
# include "RenderSettings.hh"

namespace mandelbulb {

  RenderSettings::RenderSettings(const utils::Sizef& sizeHint,
                                 sdl::core::SdlWidget* parent):
    sdl::core::SdlWidget(std::string("render_settings"),
                         sizeHint,
                         parent,
                         sdl::core::engine::Color::NamedColor::Purple)
  {
    build();
  }

  void
  RenderSettings::build() {
    // TODO: Implementation.
  }

}
