#ifndef    LIGHT_SETTINGS_HXX
# define   LIGHT_SETTINGS_HXX

# include "LightSettings.hh"

namespace mandelbulb {

  inline
  LightSettings::~LightSettings() {}

  inline
  float
  LightSettings::getGlobalMargins() noexcept {
    return 2.0f;
  }

  inline
  float
  LightSettings::getComponentMargins() noexcept {
    return 2.0f;
  }

  inline
  const char*
  LightSettings::getGeneralTextFont() noexcept {
    return "data/fonts/times.ttf";
  }

  inline
  unsigned
  LightSettings::getGeneralTextSize() noexcept {
    return 15u;
  }

  inline
  sdl::core::engine::Color
  LightSettings::getBackgroundColor() noexcept {
    return sdl::core::engine::Color::NamedColor::Indigo;
  }

  inline
  unsigned
  LightSettings::getLightsCount() noexcept {
    return 3u;
  }

  inline
  float
  LightSettings::getLabelMaxWidth() noexcept {
    return 100.0f;
  }

  inline
  std::string
  LightSettings::generateNameForLightToggle(unsigned id) noexcept {
    return std::string("light_toggle_") + std::to_string(id);
  }

  inline
  float
  LightSettings::getLightToggleButtonBordersSize() noexcept {
    return 4.0f;
  }

  inline
  sdl::graphic::Button*
  LightSettings::getButtonForLight(unsigned id) {
    return getChildAs<sdl::graphic::Button>(generateNameForLightToggle(id));
  }

  inline
  float
  LightSettings::getDefaultLightPositionCircleRadius() noexcept {
    return 4.0f;
  }

  inline
  float
  LightSettings::getDefaultLightPosition(unsigned id,
                                         char axis) noexcept
  {
    // Position the lights on a circle at a fixed `z` value using some trigonometry.
    switch (axis) {
      case 'x':
        return getDefaultLightPositionCircleRadius() * std::cos(2.0f * id * 3.1415926535f / getLightsCount());
      case 'y':
        return getDefaultLightPositionCircleRadius() * std::sin(2.0f * id * 3.1415926535f / getLightsCount());
      case 'z':
      default:
        // Assume `z` case by default.
        return 3.0f;
    }
  }

  inline
  std::string
  LightSettings::generateNameForLightPositionLabel(unsigned id,
                                                   char axis) noexcept
  {
    return std::string("light_") + axis + "_" + std::to_string(id) + "_label";
  }

  inline
  std::string
  LightSettings::generateNameForLightPositionValue(unsigned id,
                                                   char axis) noexcept
  {
    return std::string("light_") + axis + "_" + std::to_string(id) + "_value";
  }

  inline
  sdl::graphic::TextBox*
  LightSettings::getTextBoxForLightPosition(unsigned id,
                                            char axis)
  {
    return getChildAs<sdl::graphic::TextBox>(generateNameForLightPositionValue(id, axis));
  }

  inline
  float
  LightSettings::getDefaultLightPower() noexcept {
    return 10.0f;
  }

  inline
  std::string
  LightSettings::generateNameForLightPowerLabel(unsigned id) noexcept {
    return std::string("light_power_label_") + std::to_string(id);
  }

  inline
  std::string
  LightSettings::generateNameForLightPowerValue(unsigned id) noexcept {
    return std::string("light_power_value_") + std::to_string(id);
  }

  inline
  sdl::graphic::TextBox*
  LightSettings::getTextBoxForLightPower(unsigned id) {
    return getChildAs<sdl::graphic::TextBox>(generateNameForLightPowerValue(id));
  }

  inline
  std::string
  LightSettings::generateNameForLightColorLabel(unsigned id) noexcept {
    return std::string("light_color_label_") + std::to_string(id);
  }

  inline
  std::string
  LightSettings::generateNameForLightColorValue(unsigned id) noexcept {
    return std::string("light_color_value_") + std::to_string(id);
  }

  inline
  sdl::graphic::SelectorWidget*
  LightSettings::createPaletteFromIndex(unsigned id) {
    // Create the palette object.
    sdl::graphic::SelectorWidget* palette = new sdl::graphic::SelectorWidget(
      generateNameForLightColorValue(id),
      this,
      true,
      getBackgroundColor()
    );
    if (palette == nullptr) {
      error(
        std::string("Could not create light settings"),
        std::string("Could not create palette for light ") + std::to_string(id)
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
          std::string("Could not create palette entry ") + m_colors[c].toString() + " for palette " + std::to_string(id),
          utils::Level::Error
        );

        continue;
      }

      w->allowLog(false);
      palette->insertWidget(w, c);
    }

    // Assign the active color.
    palette->setActiveWidget(id);

    // Return the built-in object.
    return palette;
  }

  inline
  sdl::graphic::SelectorWidget*
  LightSettings::getPaletteForLightColor(unsigned id) {
    return getChildAs<sdl::graphic::SelectorWidget>(generateNameForLightColorValue(id));
  }

  inline
  const char*
  LightSettings::getApplyButtonName() noexcept {
    return "apply_button";
  }

  inline
  float
  LightSettings::getApplyButtonBordersSize() noexcept {
    return 4.0f;
  }

}

#endif    /* LIGHT_SETTINGS_HXX */
