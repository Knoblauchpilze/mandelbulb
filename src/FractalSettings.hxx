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
  float
  FractalSettings::getLabelMaxHeight() noexcept {
    return 50.0f;
  }

  inline
  sdl::graphic::SelectorWidget*
  FractalSettings::createPalette(const std::string& name,
                                 unsigned index)
  {
    // Create the palette object.
    sdl::graphic::SelectorWidget* palette = new sdl::graphic::SelectorWidget(
      name,
      this,
      true,
      getBackgroundColor()
    );
    if (palette == nullptr) {
      error(
        std::string("Could not create fractal settings"),
        std::string("Could not create palette \"") + name + "\""
      );
    }

    // Assign properties for the palette.
    palette->allowLog(false);

    // Populate internal colors.
    for (unsigned c = 0u ; c < m_colors.size() ; ++c) {
      sdl::core::SdlWidget* w = new sdl::core::SdlWidget(
        std::string("color_entry_") + std::to_string(c) + "_for_" + palette->getName(),
        utils::Sizef(),
        palette,
        m_colors[c]
      );
      if (w == nullptr) {
        log(
          std::string("Could not create palette entry ") + m_colors[c].toString() + " for palette " + std::to_string(index),
          utils::Level::Error
        );

        continue;
      }

      w->allowLog(false);
      palette->insertWidget(w, c);
    }

    // Assign the active color.
    palette->setActiveWidget(index);

    // Return the built-in object.
    return palette;
  }

  inline
  const char*
  FractalSettings::getFractalColorLabelName() noexcept {
    return "fractal_color_label";
  }

  inline
  const char*
  FractalSettings::getFractalColorPaletteName() noexcept {
    return "fractal_color_palette";
  }

  inline
  sdl::core::engine::Color
  FractalSettings::getDefaultFractalColor() noexcept {
    return sdl::core::engine::Color::NamedColor::Maroon;
  }

  inline
  sdl::graphic::SelectorWidget*
  FractalSettings::getFractalColorPalette() {
    return getChildAs<sdl::graphic::SelectorWidget>(getFractalColorPaletteName());
  }

  inline
  const char*
  FractalSettings::getNoDataColorLabelName() noexcept {
    return "no_data_color_label";
  }

  inline
  const char*
  FractalSettings::getNoDataColorPaletteName() noexcept {
    return "no_data_color_palette";
  }

  inline
  sdl::core::engine::Color
  FractalSettings::getDefaultNoDataColor() noexcept {
    return sdl::core::engine::Color::NamedColor::Black;
  }

  inline
  sdl::graphic::SelectorWidget*
  FractalSettings::getNoDataColorPalette() {
    return getChildAs<sdl::graphic::SelectorWidget>(getNoDataColorPaletteName());
  }

  inline
  const char*
  FractalSettings::getExposureLabelName() noexcept {
    return "exposure_label";
  }

  inline
  const char*
  FractalSettings::getExposureValueName() noexcept {
    return "exposure_value";
  }

  inline
  float
  FractalSettings::getDefaultExposureValue() noexcept {
    return 1.0f;
  }

  inline
  sdl::graphic::TextBox*
  FractalSettings::getExposureValue() {
    return getChildAs<sdl::graphic::TextBox>(getExposureValueName());
  }

  inline
  const char*
  FractalSettings::getBurnoutLabelName() noexcept {
    return "burnout_label";
  }

  inline
  const char*
  FractalSettings::getBurnoutValueName() noexcept {
    return "burnout_value";
  }

  inline
  float
  FractalSettings::getDefaultBurnoutValue() noexcept {
    return 0.1f;
  }

  inline
  sdl::graphic::TextBox*
  FractalSettings::getBurnoutValue() {
    return getChildAs<sdl::graphic::TextBox>(getBurnoutValueName());
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
