#ifndef    INFO_PANEL_HXX
# define   INFO_PANEL_HXX

# include "InfoPanel.hh"

namespace mandelbulb {

  inline
  InfoPanel::~InfoPanel() {}

  inline
  float
  InfoPanel::getGlobalMargins() noexcept {
    return 2.0f;
  }

  inline
  float
  InfoPanel::getComponentMargins() noexcept {
    return 2.0f;
  }

  inline
  const char*
  InfoPanel::getGeneralTextFont() noexcept {
    return "data/fonts/Goodtime.ttf";
  }

  inline
  unsigned
  InfoPanel::getGeneralTextSize() noexcept {
    return 15u;
  }

  inline
  sdl::core::engine::Color
  InfoPanel::getBackgroundColor() noexcept {
    return sdl::core::engine::Color::NamedColor::Indigo;
  }

  inline
  const char*
  InfoPanel::getCoordLabelName() {
    return "coord_label";
  }

  inline
  sdl::graphic::LabelWidget*
  InfoPanel::getCoordLabel() {
    return getChildAs<sdl::graphic::LabelWidget>(getCoordLabelName());
  }

  inline
  const char*
  InfoPanel::getDepthLabelName() {
    return "depth_label";
  }

  inline
  float
  InfoPanel::getDepthMaxWidth() noexcept {
    return 150.0f;
  }


  inline
  sdl::graphic::LabelWidget*
  InfoPanel::getDepthLabel() {
    return getChildAs<sdl::graphic::LabelWidget>(getDepthLabelName());
  }

  inline
  const char*
  InfoPanel::getCameraLabelName() {
    return "camera_label";
  }

  inline
  sdl::graphic::LabelWidget*
  InfoPanel::getCameraLabel() {
    return getChildAs<sdl::graphic::LabelWidget>(getCameraLabelName());
  }

}

#endif    /* INFO_PANEL_HXX */
