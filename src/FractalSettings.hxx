#ifndef    FRACTAL_SETTINGS_HXX
# define   FRACTAL_SETTINGS_HXX

# include "FractalSettings.hh"

namespace mandelbulb {

  inline
  FractalSettings::~FractalSettings() {}

  inline
  float
  FractalSettings::getGlobalMargins() noexcept {
    return 2.0f;
  }

  inline
  float
  FractalSettings::getComponentMargins() noexcept {
    return 2.0f;
  }

  inline
  const char*
  FractalSettings::getGeneralTextFont() noexcept {
    return "data/fonts/times.ttf";
  }

  inline
  unsigned
  FractalSettings::getGeneralTextSize() noexcept {
    return 15u;
  }

  inline
  sdl::core::engine::Color
  FractalSettings::getBackgroundColor() noexcept {
    return sdl::core::engine::Color::NamedColor::Indigo;
  }

  inline
  const char*
  FractalSettings::getApplyButtonName() noexcept {
    return "apply_button";
  }

  inline
  float
  FractalSettings::getApplyButtonBordersSize() noexcept {
    return 4.0f;
  }

}

#endif    /* FRACTAL_SETTINGS_HXX */
