
# include "LightSettings.hh"
# include <sstream>
# include <iomanip>
# include <sdl_graphic/GridLayout.hh>
# include <core_utils/Conversion.hh>

namespace mandelbulb {

  LightSettings::LightSettings(const utils::Sizef& sizeHint,
                               sdl::core::SdlWidget* parent):
    sdl::core::SdlWidget(std::string("light_settings"),
                         sizeHint,
                         parent,
                         getBackgroundColor()),

    m_colors(),

    onLightsChanged()
  {
    build();
  }

  void
  LightSettings::build() {
    // Create the palette to be used when creating color panels for lights.
    generatePalette();

    // This component uses a grid layout to position the internal components.
    // Each light panel contains:
    //  - a toggle button to de/activate it.
    //  - three lines for its position.
    //  - a line for its power.
    //  - a line for its color.
    // So a grand total of `6` values for each light panel.
    const unsigned lightPanelSize = 6u;

    sdl::graphic::GridLayoutShPtr layout = std::make_shared<sdl::graphic::GridLayout>(
      "info_panel_layout",
      this,
      2u,
      1u + getLightsCount() * lightPanelSize,
      getGlobalMargins()
    );

    layout->allowLog(false);

    setLayout(layout);

    // Create each light panel. A light panel is composed of a toggle
    // button allowing to de/activate the light, of some options to
    // configure its position and power and its color.
    std::string x("x:"), y("y:"), z("z:"), power("Power:"), color("Color:");
    std::stringstream ss;

    for (unsigned id = 0u ; id < getLightsCount() ; ++id) {
      // Create the toggle button.
      sdl::graphic::Button* toggle = new sdl::graphic::Button(
        generateNameForLightToggle(id),
        std::string("Toggle"),
        std::string(),
        getGeneralTextFont(),
        sdl::graphic::button::Type::Toggle,
        getGeneralTextSize(),
        this,
        getLightToggleButtonBordersSize()
      );
      if (toggle == nullptr) {
        error(
          std::string("Could not create light settings"),
          std::string("Toggle button for light ") + std::to_string(id) + " not created"
        );
      }

      // Create the labels and textboxes to enter the position of the
      // light.
      sdl::graphic::LabelWidget* xLabel = new sdl::graphic::LabelWidget(
        generateNameForLightPositionLabel(id, 'x'),
        x,
        getGeneralTextFont(),
        getGeneralTextSize(),
        sdl::graphic::LabelWidget::HorizontalAlignment::Left,
        sdl::graphic::LabelWidget::VerticalAlignment::Center,
        this,
        getBackgroundColor()
      );
      if (xLabel == nullptr) {
        error(
          std::string("Could not create light settings"),
          std::string("Label for x coordinate of light ") + std::to_string(id) + " not created"
        );
      }

      ss.clear();
      ss.str("");
      ss << std::setprecision(2) << getDefaultLightPosition(id, 'x');
      sdl::graphic::TextBox* xValue = new sdl::graphic::TextBox(
        generateNameForLightPositionValue(id, 'x'),
        getGeneralTextFont(),
        ss.str(),
        getGeneralTextSize(),
        this
      );
      if (xValue == nullptr) {
        error(
          std::string("Could not create light settings"),
          std::string("Value for x coordinate of light ") + std::to_string(id) + " not created"
        );
      }

      sdl::graphic::LabelWidget* yLabel = new sdl::graphic::LabelWidget(
        generateNameForLightPositionLabel(id, 'y'),
        y,
        getGeneralTextFont(),
        getGeneralTextSize(),
        sdl::graphic::LabelWidget::HorizontalAlignment::Left,
        sdl::graphic::LabelWidget::VerticalAlignment::Center,
        this,
        getBackgroundColor()
      );
      if (yLabel == nullptr) {
        error(
          std::string("Could not create light settings"),
          std::string("Label for y coordinate of light ") + std::to_string(id) + " not created"
        );
      }

      ss.clear();
      ss.str("");
      ss << std::setprecision(2) << getDefaultLightPosition(id, 'y');
      sdl::graphic::TextBox* yValue = new sdl::graphic::TextBox(
        generateNameForLightPositionValue(id, 'y'),
        getGeneralTextFont(),
        ss.str(),
        getGeneralTextSize(),
        this
      );
      if (yValue == nullptr) {
        error(
          std::string("Could not create light settings"),
          std::string("Value for y coordinate of light ") + std::to_string(id) + " not created"
        );
      }

      sdl::graphic::LabelWidget* zLabel = new sdl::graphic::LabelWidget(
        generateNameForLightPositionLabel(id, 'z'),
        z,
        getGeneralTextFont(),
        getGeneralTextSize(),
        sdl::graphic::LabelWidget::HorizontalAlignment::Left,
        sdl::graphic::LabelWidget::VerticalAlignment::Center,
        this,
        getBackgroundColor()
      );
      if (zLabel == nullptr) {
        error(
          std::string("Could not create light settings"),
          std::string("Label for z coordinate of light ") + std::to_string(id) + " not created"
        );
      }

      ss.clear();
      ss.str("");
      ss << std::setprecision(2) << getDefaultLightPosition(id, 'z');
      sdl::graphic::TextBox* zValue = new sdl::graphic::TextBox(
        generateNameForLightPositionValue(id, 'z'),
        getGeneralTextFont(),
        ss.str(),
        getGeneralTextSize(),
        this
      );
      if (zValue == nullptr) {
        error(
          std::string("Could not create light settings"),
          std::string("Value for z coordinate of light ") + std::to_string(id) + " not created"
        );
      }

      // Create the label and textbox to enter the power of the light.
      sdl::graphic::LabelWidget* powerLabel = new sdl::graphic::LabelWidget(
        generateNameForLightPowerLabel(id),
        power,
        getGeneralTextFont(),
        getGeneralTextSize(),
        sdl::graphic::LabelWidget::HorizontalAlignment::Left,
        sdl::graphic::LabelWidget::VerticalAlignment::Center,
        this,
        getBackgroundColor()
      );
      if (powerLabel == nullptr) {
        error(
          std::string("Could not create light settings"),
          std::string("Power label for light ") + std::to_string(id) + " not created"
        );
      }

      ss.clear();
      ss.str("");
      ss << std::setprecision(2) << getDefaultLightPower();
      sdl::graphic::TextBox* powerValue = new sdl::graphic::TextBox(
        generateNameForLightPowerValue(id),
        getGeneralTextFont(),
        ss.str(),
        getGeneralTextSize(),
        this
      );
      if (powerValue == nullptr) {
        error(
          std::string("Could not create light settings"),
          std::string("Power value for light ") + std::to_string(id) + " not created"
        );
      }

      // Create the label and palettes to enter the color of the light.
      sdl::graphic::LabelWidget* colorLabel = new sdl::graphic::LabelWidget(
        generateNameForLightColorLabel(id),
        color,
        getGeneralTextFont(),
        getGeneralTextSize(),
        sdl::graphic::LabelWidget::HorizontalAlignment::Left,
        sdl::graphic::LabelWidget::VerticalAlignment::Center,
        this,
        getBackgroundColor()
      );
      if (colorLabel == nullptr) {
        error(
          std::string("Could not create light settings"),
          std::string("Color label for light ") + std::to_string(id) + " not created"
        );
      }

      sdl::graphic::SelectorWidget* colorValue = createPaletteFromIndex(id);
      if (colorValue == nullptr) {
        error(
          std::string("Could not create light settings"),
          std::string("Color value for light ") + std::to_string(id) + " not created"
        );
      }

      // Apply size restrictions to each component.
      utils::Sizef maxSz(getLabelMaxWidth(), std::numeric_limits<float>::max());

      xLabel->setMaxSize(maxSz);
      yLabel->setMaxSize(maxSz);
      zLabel->setMaxSize(maxSz);
      powerLabel->setMaxSize(maxSz);
      colorLabel->setMaxSize(maxSz);

      toggle->allowLog(false);
      xLabel->allowLog(false);
      xValue->allowLog(false);
      yLabel->allowLog(false);
      yValue->allowLog(false);
      zLabel->allowLog(false);
      zValue->allowLog(false);
      powerLabel->allowLog(false);
      powerValue->allowLog(false);
      colorLabel->allowLog(false);
      colorValue->allowLog(false);

      // Append each item to the layout.
      layout->addItem(toggle,     0u, id * lightPanelSize + 0u, 2u, 1u);
      layout->addItem(xLabel,     0u, id * lightPanelSize + 1u, 1u, 1u);
      layout->addItem(xValue,     1u, id * lightPanelSize + 1u, 1u, 1u);
      layout->addItem(yLabel,     0u, id * lightPanelSize + 2u, 1u, 1u);
      layout->addItem(yValue,     1u, id * lightPanelSize + 2u, 1u, 1u);
      layout->addItem(zLabel,     0u, id * lightPanelSize + 3u, 1u, 1u);
      layout->addItem(zValue,     1u, id * lightPanelSize + 3u, 1u, 1u);
      layout->addItem(powerLabel, 0u, id * lightPanelSize + 4u, 1u, 1u);
      layout->addItem(powerValue, 1u, id * lightPanelSize + 4u, 1u, 1u);
      layout->addItem(colorLabel, 0u, id * lightPanelSize + 5u, 1u, 1u);
      layout->addItem(colorValue, 1u, id * lightPanelSize + 5u, 1u, 1u);
    }

    // Create the apply button data.
    sdl::graphic::Button* apply = new sdl::graphic::Button(
      getApplyButtonName(),
      std::string("Apply"),
      std::string(),
      getGeneralTextFont(),
      sdl::graphic::button::Type::Regular,
      getGeneralTextSize(),
      this,
      getApplyButtonBordersSize(),
      utils::Sizef(),
      sdl::core::engine::Color::NamedColor::Teal
    );
    if (apply == nullptr) {
      error(
        std::string("Could not create light settings"),
        std::string("Apply button not created")
      );
    }

    // Build layout with the `apply` button (the rest is already built).
    layout->addItem(apply, 0u, getLightsCount() * lightPanelSize, 2u, 1u);

    // Connect the button `onClick` signal to the local slot in order to
    // be able to propagate the lights' properties to external listeners.
    apply->onClick.connect_member<LightSettings>(
      this,
      &LightSettings::onApplyButtonClicked
    );
  }

  void
  LightSettings::generatePalette() noexcept {
    // Generate palette in order they are defined.
    m_colors.push_back(sdl::core::engine::Color::NamedColor::White);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Black);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Red);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Green);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Blue);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Yellow);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Orange);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Cyan);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Magenta);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Silver);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Gray);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Maroon);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Olive);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Pink);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Purple);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Teal);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Navy);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Indigo);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::CorneFlowerBlue);
  }

  void
  LightSettings::onApplyButtonClicked(const std::string& /*dummy*/) {
    // We need to traverse the internal elements to fetch all the lights
    // and build the corresponding data vector.
    std::vector<LightShPtr> lights;
    bool success = false;

    for (unsigned id = 0u ; id < getLightsCount() ; ++id) {
      // Retrieve the properties for the current light if any.
      sdl::graphic::Button* toggle = getButtonForLight(id);
      if (toggle == nullptr) {
        log("Could not retrieve palette for light " + std::to_string(id), utils::Level::Error);
        continue;
      }

      if (!toggle->toggled()) {
        // The light is not active, do not account for it.
        continue;
      }

      // Retrieve the position of the light.
      sdl::graphic::TextBox* xTB = getTextBoxForLightPosition(id, 'x');
      sdl::graphic::TextBox* yTB = getTextBoxForLightPosition(id, 'y');
      sdl::graphic::TextBox* zTB = getTextBoxForLightPosition(id, 'z');

      if (xTB == nullptr) {
        log("Could not retrieve x coordinate for light " + std::to_string(id), utils::Level::Error);
        continue;
      }
      if (yTB == nullptr) {
        log("Could not retrieve y coordinate for light " + std::to_string(id), utils::Level::Error);
        continue;
      }
      if (zTB == nullptr) {
        log("Could not retrieve z coordinate for light " + std::to_string(id), utils::Level::Error);
        continue;
      }

      std::string xStr, yStr, zStr;
      withSafetyNet(
        [&xStr, &yStr, &zStr, xTB, yTB, zTB]() {
          xStr = xTB->getValue();
          yStr = yTB->getValue();
          zStr = zTB->getValue();
        },
        std::string("Light_") + std::to_string(id) + "::getPos()"
      );

      utils::Vector3f pos(
        getDefaultLightPosition(id, 'x'),
        getDefaultLightPosition(id, 'y'),
        getDefaultLightPosition(id, 'z')
      );

      pos.x() = utils::convert(xStr, getDefaultLightPosition(id, 'x'), success);
      if (!success) {
        log(
          std::string("Could not convert provided x coordinate of \"") + xStr + "\" for light " +
          std::to_string(id) + " using " + std::to_string(getDefaultLightPosition(id, 'x')) + " instead",
          utils::Level::Warning
        );
      }
      pos.y() = utils::convert(yStr, getDefaultLightPosition(id, 'y'), success);
      if (!success) {
        log(
          std::string("Could not convert provided y coordinate of \"") + yStr + "\" for light " +
          std::to_string(id) + " using " + std::to_string(getDefaultLightPosition(id, 'y')) + " instead",
          utils::Level::Warning
        );
      }
      pos.z() = utils::convert(zStr, getDefaultLightPosition(id, 'z'), success);
      if (!success) {
        log(
          std::string("Could not convert provided z coordinate of \"") + zStr + "\" for light " +
          std::to_string(id) + " using " + std::to_string(getDefaultLightPosition(id, 'z')) + " instead",
          utils::Level::Warning
        );
      }

      // Retrieve the color.
      sdl::graphic::SelectorWidget* palette = getPaletteForLightColor(id);
      if (palette == nullptr) {
        log("Could not retrieve palette for light " + std::to_string(id), utils::Level::Error);
        continue;
      }

      // Populate the color based on the current active on in the palette
      // and by checking the assocation in the internal colors table.
      unsigned cID = static_cast<unsigned>(palette->getActiveItem());
      if (cID >= m_colors.size()) {
        log(
          std::string("Cannot associate color ") + std::to_string(cID) + " for light " +
          std::to_string(id) + ", palette only defines " + std::to_string(m_colors.size()),
          utils::Level::Error
        );

        continue;
      }

      sdl::core::engine::Color c = m_colors[cID];

      // Retrieve the intensity.
      sdl::graphic::TextBox* powerTB = getTextBoxForLightPower(id);
      if (powerTB == nullptr) {
        log("Could not retrieve intensity for light " + std::to_string(id), utils::Level::Error);
        continue;
      }

      std::string powerStr;
      withSafetyNet(
        [&powerStr, powerTB]() {
          powerStr = powerTB->getValue();
        },
        std::string("Light_") + std::to_string(id) + "::getPower()"
      );

      float intensity = utils::convert(powerStr, getDefaultLightPower(), success);
      if (!success) {
        log(
          std::string("Could not convert provided intensity of \"") + powerStr + "\" for light " +
          std::to_string(id) + " using " + std::to_string(intensity) + " instead",
          utils::Level::Warning
        );
      }

      // Create the light and register it for the output signal.
      LightShPtr light = Light::fromPositionAndTarget(pos, utils::Vector3f(0.0f, 0.0f, 0.0f));
      light->setColor(c);
      light->setIntensity(intensity);

      lights.push_back(light);
    }

    // Notify listeners.
    onLightsChanged.safeEmit(
      std::string("onLightsChanged(") + std::to_string(lights.size()) + ")",
      lights
    );
  }

}
