#ifndef    RENDER_SETTINGS_HXX
# define   RENDER_SETTINGS_HXX

# include "RenderSettings.hh"

namespace mandelbulb {

  inline
  RenderSettings::~RenderSettings() {}

  inline
  float
  RenderSettings::getGlobalMargins() noexcept {
    return 2.0f;
  }

  inline
  float
  RenderSettings::getComponentMargins() noexcept {
    return 2.0f;
  }

  inline
  const char*
  RenderSettings::getGeneralTextFont() noexcept {
    return "data/fonts/times.ttf";
  }

  inline
  unsigned
  RenderSettings::getGeneralTextSize() noexcept {
    return 15u;
  }

  inline
  sdl::core::engine::Color
  RenderSettings::getBackgroundColor() noexcept {
    return sdl::core::engine::Color::NamedColor::Indigo;
  }

  inline
  float
  RenderSettings::getDisplayElementMaxHeight() noexcept {
    return 60.0f;
  }

  inline
  const char*
  RenderSettings::getPowerLabelName() noexcept {
    return "power_label";
  }

  inline
  const char*
  RenderSettings::getPowerTextBoxName() noexcept {
    return "power_value";
  }

  inline
  float
  RenderSettings::getDefaultPower() noexcept {
    return 8.0f;
  }

  inline
  sdl::graphic::TextBox*
  RenderSettings::getPowerValueTextBox() {
    return getChildAs<sdl::graphic::TextBox>(getPowerTextBoxName());
  }

  inline
  const char*
  RenderSettings::getAccuracyLabelName() noexcept {
    return "accuracy_label";
  }

  inline
  const char*
  RenderSettings::getAccuracyTextBoxName() noexcept {
    return "accuracy_value";
  }

  inline
  unsigned
  RenderSettings::getDefaultAccuracy() noexcept {
    return 128u;
  }

  inline
  unsigned
  RenderSettings::getDefaultMaxIterations() noexcept {
    return 50u;
  }

  sdl::graphic::TextBox*
  RenderSettings::getAccuracyTextBox() {
    return getChildAs<sdl::graphic::TextBox>(getAccuracyTextBoxName());
  }

  inline
  const char*
  RenderSettings::getMaxIterationsLabelName() noexcept {
    return "max_iter_label";
  }

  inline
  const char*
  RenderSettings::getMaxIterationsTextBoxName() noexcept {
    return "max_iter_value";
  }

  inline
  sdl::graphic::TextBox*
  RenderSettings::getMaxIterationsTextBox() {
    return getChildAs<sdl::graphic::TextBox>(getMaxIterationsTextBoxName());
  }

  inline
  const char*
  RenderSettings::getApplyButtonName() noexcept {
    return "apply_button";
  }

  inline
  float
  RenderSettings::getApplyButtonBordersSize() noexcept {
    return 4.0f;
  }

}

#endif    /* RENDER_SETTINGS_HXX */
