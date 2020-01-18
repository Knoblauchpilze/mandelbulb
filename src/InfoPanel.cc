
# include "InfoPanel.hh"

namespace mandelbulb {

  InfoPanel::InfoPanel(const utils::Sizef& sizeHint,
                       sdl::core::SdlWidget* parent):
    sdl::core::SdlWidget(std::string("info_panel"),
                         sizeHint,
                         parent,
                         sdl::core::engine::Color::NamedColor::Purple)
  {
    build();
  }

  void
  InfoPanel::build() {
    // TODO: Implementation.
  }

}
