
# include "LightSettings.hh"

namespace mandelbulb {

  LightSettings::LightSettings(const utils::Sizef& sizeHint,
                               sdl::core::SdlWidget* parent):
    sdl::core::SdlWidget(std::string("light_settings"),
                         sizeHint,
                         parent,
                         sdl::core::engine::Color::NamedColor::Purple)
  {
    build();
  }

  void
  LightSettings::build() {
    // TODO: Implementation.
  }

}
