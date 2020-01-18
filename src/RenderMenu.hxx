#ifndef    RENDER_MENU_HXX
# define   RENDER_MENU_HXX

# include "RenderMenu.hh"

namespace mandelbulb {

  inline
  RenderMenu::~RenderMenu() {}

  inline
  float
  RenderMenu::getGlobalMargins() noexcept {
    return 2.0f;
  }

  inline
  float
  RenderMenu::getComponentMargins() noexcept {
    return 2.0f;
  }

  inline
  const char*
  RenderMenu::getGeneralTextFont() noexcept {
    return "data/fonts/Goodtime.ttf";
  }

  inline
  unsigned
  RenderMenu::getGeneralTextSize() noexcept {
    return 15u;
  }

  inline
  sdl::core::engine::Color
  RenderMenu::getBackgroundColor() noexcept {
    return sdl::core::engine::Color::NamedColor::Indigo;
  }

  inline
  float
  RenderMenu::getPercentageTextMaxWidth() noexcept {
    return 150.0f;
  }

  inline
  const char*
  RenderMenu::getProgressBarName() noexcept {
    return "progress_bar";
  }

  inline
  sdl::graphic::ProgressBar*
  RenderMenu::getProgressBar() {
    return getChildAs<sdl::graphic::ProgressBar>(getProgressBarName());
  }

  inline
  const char*
  RenderMenu::getPercentageLabelName() noexcept {
    return "percentage_label";
  }

  inline
  sdl::graphic::LabelWidget*
  RenderMenu::getPercentageLabel() {
    return getChildAs<sdl::graphic::LabelWidget>(getPercentageLabelName());
  }

}

#endif    /* RENDER_MENU_HXX */
