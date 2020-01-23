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
  constexpr float
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
  constexpr unsigned
  RenderSettings::getDefaultAccuracy() noexcept {
    return 128u;
  }

  sdl::graphic::TextBox*
  RenderSettings::getAccuracyTextBox() {
    return getChildAs<sdl::graphic::TextBox>(getAccuracyTextBoxName());
  }

  inline
  const char*
  RenderSettings::getHitThresholdLabelName() noexcept {
    return "hit_thresh_label";
  }

  inline
  const char*
  RenderSettings::getHitThresholdTextBoxName() noexcept {
    return "hit_thresh_value";
  }

  inline
  constexpr float
  RenderSettings::getDefaultHitThreshold() noexcept {
    return 0.01f;
  }

  inline
  sdl::graphic::TextBox*
  RenderSettings::getHitThresholdTextBox() {
    return getChildAs<sdl::graphic::TextBox>(getHitThresholdTextBoxName());
  }

  inline
  const char*
  RenderSettings::getRayStepsLabelName() noexcept {
    return "ray_steps_label";
  }

  inline
  const char*
  RenderSettings::getRayStepsTextBoxName() noexcept {
    return "ray_steps_value";
  }

  inline
  constexpr unsigned
  RenderSettings::getDefaultRaySteps() noexcept {
    return 20u;
  }

  inline
  sdl::graphic::TextBox*
  RenderSettings::getRayStepsTextBox() {
    return getChildAs<sdl::graphic::TextBox>(getRayStepsTextBoxName());
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
  constexpr unsigned
  RenderSettings::getDefaultMaxIterations() noexcept {
    return 10u;
  }

  inline
  const char*
  RenderSettings::getBailoutLabelName() noexcept {
    return "bailout_label";
  }

  inline
  const char*
  RenderSettings::getBailoutTextBoxName() noexcept {
    return "bailout_value";
  }

  inline
  float
  RenderSettings::getDefaultBailout() noexcept {
    return 4.0f;
  }

  inline
  sdl::graphic::TextBox*
  RenderSettings::getBailoutTextBox() {
    return getChildAs<sdl::graphic::TextBox>(getBailoutTextBoxName());
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
