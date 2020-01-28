
# include "FractalSettings.hh"
# include <sstream>
# include <iomanip>
# include <sdl_graphic/GridLayout.hh>
# include <sdl_graphic/LabelWidget.hh>
# include <sdl_graphic/Button.hh>
# include <core_utils/Conversion.hh>

namespace mandelbulb {

  FractalSettings::FractalSettings(const utils::Sizef& sizeHint,
                                   sdl::core::SdlWidget* parent):
    sdl::core::SdlWidget(std::string("fractal_settings"),
                         sizeHint,
                         parent,
                         getBackgroundColor()),

    m_propsLocker(),
    m_colors(),

    onShadingPropertiesChanged()
  {
    build();
  }

  void
  FractalSettings::build() {
    // Create the palette to be used when creating color panels for lights.
    generatePalette();

    // This component uses a grid layout to position the internal components.
    sdl::graphic::GridLayoutShPtr layout = std::make_shared<sdl::graphic::GridLayout>(
      "fractal_settings_layout",
      this,
      1u,
      9u,
      getGlobalMargins()
    );

    layout->allowLog(false);

    setLayout(layout);

    std::stringstream ss;

    // Fractal color data.
    sdl::graphic::LabelWidget* fColorLabel = new sdl::graphic::LabelWidget(
      getFractalColorLabelName(),
      std::string("Fractal:"),
      getGeneralTextFont(),
      getGeneralTextSize(),
      sdl::graphic::LabelWidget::HorizontalAlignment::Left,
      sdl::graphic::LabelWidget::VerticalAlignment::Center,
      this,
      getBackgroundColor()
    );
    if (fColorLabel == nullptr) {
      error(
        std::string("Could not create fractal settings"),
        std::string("Fractal color label not created")
      );
    }

    sdl::graphic::SelectorWidget* fColorValue = createPalette(getFractalColorPaletteName(), 13u);
    if (fColorValue == nullptr) {
      error(
        std::string("Could not create fractal settings"),
        std::string("Fractal color palette not created")
      );
    }

    // No data color elements.
    sdl::graphic::LabelWidget* ndColorLabel = new sdl::graphic::LabelWidget(
      getNoDataColorLabelName(),
      std::string("No data:"),
      getGeneralTextFont(),
      getGeneralTextSize(),
      sdl::graphic::LabelWidget::HorizontalAlignment::Left,
      sdl::graphic::LabelWidget::VerticalAlignment::Center,
      this,
      getBackgroundColor()
    );
    if (ndColorLabel == nullptr) {
      error(
        std::string("Could not create fractal settings"),
        std::string("No data color label not created")
      );
    }

    sdl::graphic::SelectorWidget* ndColorValue = createPalette(getNoDataColorPaletteName(), 5u);
    if (ndColorValue == nullptr) {
      error(
        std::string("Could not create fractal settings"),
        std::string("No data color palette not created")
      );
    }

    // Tonemap exposure.
    sdl::graphic::LabelWidget* exposureLabel = new sdl::graphic::LabelWidget(
      getExposureLabelName(),
      std::string("Exposure:"),
      getGeneralTextFont(),
      getGeneralTextSize(),
      sdl::graphic::LabelWidget::HorizontalAlignment::Left,
      sdl::graphic::LabelWidget::VerticalAlignment::Center,
      this,
      getBackgroundColor()
    );
    if (exposureLabel == nullptr) {
      error(
        std::string("Could not create fractal settings"),
        std::string("Tonemap exposure label not created")
      );
    }

    ss << std::setprecision(0) << getDefaultExposureValue();

    sdl::graphic::TextBox* exposureValue = new sdl::graphic::TextBox(
      getExposureValueName(),
      getGeneralTextFont(),
      ss.str(),
      getGeneralTextSize(),
      this
    );
    if (exposureValue == nullptr) {
      error(
        std::string("Could not create fractal settings"),
        std::string("Tonemap exposure textbox not created")
      );
    }

    // Tonemap burnout.
    sdl::graphic::LabelWidget* burnoutLabel = new sdl::graphic::LabelWidget(
      getBurnoutLabelName(),
      std::string("Burnout:"),
      getGeneralTextFont(),
      getGeneralTextSize(),
      sdl::graphic::LabelWidget::HorizontalAlignment::Left,
      sdl::graphic::LabelWidget::VerticalAlignment::Center,
      this,
      getBackgroundColor()
    );
    if (exposureLabel == nullptr) {
      error(
        std::string("Could not create fractal settings"),
        std::string("Tonemap burnout label not created")
      );
    }

    ss.clear();
    ss.str("");
    ss << std::setprecision(2) << getDefaultBurnoutValue();

    sdl::graphic::TextBox* burnoutValue = new sdl::graphic::TextBox(
      getBurnoutValueName(),
      getGeneralTextFont(),
      ss.str(),
      getGeneralTextSize(),
      this
    );
    if (burnoutValue == nullptr) {
      error(
        std::string("Could not create fractal settings"),
        std::string("Tonemap burnout textbox not created")
      );
    }

    // Apply button data.
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
        std::string("Could not create fractal settings"),
        std::string("Apply button not created")
      );
    }

    // Apply size restrictions to each component.
    utils::Sizef maxSz(std::numeric_limits<float>::max(), getLabelMaxHeight());

    fColorLabel->setMaxSize(maxSz);
    fColorValue->setMaxSize(maxSz);
    ndColorLabel->setMaxSize(maxSz);
    ndColorValue->setMaxSize(maxSz);
    exposureLabel->setMaxSize(maxSz);
    exposureValue->setMaxSize(maxSz);
    burnoutLabel->setMaxSize(maxSz);
    burnoutValue->setMaxSize(maxSz);
    apply->setMaxSize(maxSz);

    fColorLabel->allowLog(false);
    fColorValue->allowLog(false);
    ndColorLabel->allowLog(false);
    ndColorValue->allowLog(false);
    exposureLabel->allowLog(false);
    exposureValue->allowLog(false);
    burnoutLabel->allowLog(false);
    burnoutValue->allowLog(false);
    apply->allowLog(false);

    // Build layout for this component.
    layout->addItem(fColorLabel,   0u, 0u, 1u, 1u);
    layout->addItem(fColorValue,   0u, 1u, 1u, 1u);
    layout->addItem(ndColorLabel,  0u, 2u, 1u, 1u);
    layout->addItem(ndColorValue,  0u, 3u, 1u, 1u);
    layout->addItem(exposureLabel, 0u, 4u, 1u, 1u);
    layout->addItem(exposureValue, 0u, 5u, 1u, 1u);
    layout->addItem(burnoutLabel,  0u, 6u, 1u, 1u);
    layout->addItem(burnoutValue,  0u, 7u, 1u, 1u);
    layout->addItem(apply,         0u, 8u, 1u, 1u);

    // Connect the button `onClick` signal to the local slot in order to
    // be able to propagate the lights' properties to external listeners.
    apply->onClick.connect_member<FractalSettings>(
      this,
      &FractalSettings::onApplyButtonClicked
    );
  }

  void
  FractalSettings::generatePalette() noexcept {
    // Generate a palette. We will use five pre-defined colors and
    // then add all named colors in order they are defined.
    m_colors.push_back(sdl::core::engine::Color::NamedColor::White);
    m_colors.push_back(sdl::core::engine::Color::fromRGB(0.7f, 0.7f, 0.3f));
    m_colors.push_back(sdl::core::engine::Color::fromRGB(0.0f, 0.5f, 1.0f));
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Orange);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Green);

    m_colors.push_back(sdl::core::engine::Color::NamedColor::Black);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Red);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Blue);
    m_colors.push_back(sdl::core::engine::Color::NamedColor::Yellow);
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
  FractalSettings::onApplyButtonClicked(const std::string& /*dummy*/) {
    // Retrieve properties from the internal components.
    // Fractal color.
    sdl::core::engine::Color fColor = getDefaultFractalColor();

    sdl::graphic::SelectorWidget* fPalette = getFractalColorPalette();
    if (fPalette == nullptr) {
      log(
        std::string("Could not retrieve fractal color palette, using default color ") + fColor.toString(),
        utils::Level::Error
      );
    }
    else {
      unsigned cID = static_cast<unsigned>(fPalette->getActiveItem());
      if (cID >= m_colors.size()) {
        log(
          std::string("Cannot associate fractal color ") + std::to_string(cID) +
          ", palette only defines " + std::to_string(m_colors.size()) + ", using " + fColor.toString(),
          utils::Level::Error
        );
      }
      else {
        fColor = m_colors[cID];
      }
    }

    // No data color.
    sdl::core::engine::Color ndColor = getDefaultNoDataColor();

    sdl::graphic::SelectorWidget* ndPalette = getNoDataColorPalette();
    if (ndPalette == nullptr) {
      log(
        std::string("Could not retrieve no data color palette, using default color ") + ndColor.toString(),
        utils::Level::Error
      );
    }
    else {
      unsigned cID = static_cast<unsigned>(ndPalette->getActiveItem());
      if (cID >= m_colors.size()) {
        log(
          std::string("Cannot associate no data color ") + std::to_string(cID) +
          ", palette only defines " + std::to_string(m_colors.size()) + ", using " + ndColor.toString(),
          utils::Level::Error
        );
      }
      else {
        ndColor = m_colors[cID];
      }
    }

    // Exposure value.
    float exposure = getDefaultExposureValue();

    sdl::graphic::TextBox* expTB = getExposureValue();
    if (expTB == nullptr) {
      log(
        std::string("Could not retrieve exposure textbox, using default value ") + std::to_string(exposure),
        utils::Level::Error
      );
    }
    else {
      std::string expStr;
      withSafetyNet(
        [&expStr, expTB]() {
          expStr = expTB->getValue();
        },
        std::string("shading::getExposure()")
      );

      bool success;
      exposure = utils::convert(expStr, getDefaultExposureValue(), success);

      if (!success) {
        log(
          std::string("Could not convert exposure value \"") + expStr + "\", using default value " + std::to_string(exposure),
          utils::Level::Error
        );
      }
    }

    // Burnout value.
    float burnout = getDefaultBurnoutValue();

    sdl::graphic::TextBox* burnTB = getBurnoutValue();
    if (burnTB == nullptr) {
      log(
        std::string("Could not retrieve burnout textbox, using default value ") + std::to_string(burnout),
        utils::Level::Error
      );
    }
    else {
      std::string burnStr;
      withSafetyNet(
        [&burnStr, burnTB]() {
          burnStr = burnTB->getValue();
        },
        std::string("shading::getBurnout()")
      );

      bool success;
      burnout = utils::convert(burnStr, getDefaultBurnoutValue(), success);

      if (!success) {
        log(
          std::string("Could not convert burnout value \"") + burnStr + "\", using default value " + std::to_string(burnout),
          utils::Level::Error
        );
      }
    }

    // Build the shading properties object.
    ShadingProperties props{
      fColor, ndColor,
      exposure, burnout
    };

    // Notify external listeners.
    Guard guard(m_propsLocker);

    onShadingPropertiesChanged.safeEmit(
      std::string("onShadingPropertiesChanged()"),
      props
    );
  }

}
