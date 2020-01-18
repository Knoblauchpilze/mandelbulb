
# include "RenderMenu.hh"

namespace mandelbulb {

  RenderMenu::RenderMenu(const utils::Sizef& sizeHint,
                         sdl::core::SdlWidget* parent):
    sdl::core::SdlWidget(std::string("render_menu"),
                         sizeHint,
                         parent,
                         sdl::core::engine::Color::NamedColor::Purple)
  {
    build();
  }

  void
  RenderMenu::build() {
    // TODO: Implementation.
  }

}
