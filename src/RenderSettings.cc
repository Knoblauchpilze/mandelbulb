
# include "RenderSettings.hh"
# include <sstream>
# include <iomanip>
# include <sdl_graphic/LinearLayout.hh>
# include <sdl_graphic/Button.hh>
# include <core_utils/Conversion.hh>

namespace mandelbulb {

  RenderSettings::RenderSettings(const utils::Sizef& sizeHint,
                                 sdl::core::SdlWidget* parent):
    sdl::core::SdlWidget(std::string("render_settings"),
                         sizeHint,
                         parent,
                         getBackgroundColor()),

    m_propsLocker(),

    onRenderingSettingsChanged()
  {
    build();
  }

  void
  RenderSettings::build() {
    // This component uses a vertical linear layout to position the
    // internal components.
    sdl::graphic::LinearLayoutShPtr layout = std::make_shared<sdl::graphic::LinearLayout>(
      "render_settings_layout",
      this,
      sdl::graphic::LinearLayout::Direction::Vertical,
      getGlobalMargins(),
      getComponentMargins()
    );

    layout->allowLog(false);

    setLayout(layout);

    utils::Sizef szMax(
      std::numeric_limits<float>::max(),
      getDisplayElementMaxHeight()
    );

    // Create power data.
    sdl::graphic::LabelWidget* powerLabel = new sdl::graphic::LabelWidget(
      getPowerLabelName(),
      std::string("Power:"),
      getGeneralTextFont(),
      getGeneralTextSize(),
      sdl::graphic::LabelWidget::HorizontalAlignment::Left,
      sdl::graphic::LabelWidget::VerticalAlignment::Center,
      this,
      getBackgroundColor()
    );
    if (powerLabel == nullptr) {
      error(
        std::string("Could not create render settings"),
        std::string("Power label not created")
      );
    }

    powerLabel->setMaxSize(szMax);

    std::stringstream ss;
    ss << std::setprecision(1) << getDefaultPower();

    sdl::graphic::TextBox* powerValue = new sdl::graphic::TextBox(
      getPowerTextBoxName(),
      getGeneralTextFont(),
      ss.str(),
      getGeneralTextSize(),this
    );
    if (powerValue == nullptr) {
      error(
        std::string("Could not create render settings"),
        std::string("Power textbox not created")
      );
    }

    powerValue->setMaxSize(szMax);

    // Create accuracy data.
    sdl::graphic::LabelWidget* accuracyLabel = new sdl::graphic::LabelWidget(
      getAccuracyLabelName(),
      std::string("Accuracy:"),
      getGeneralTextFont(),
      getGeneralTextSize(),
      sdl::graphic::LabelWidget::HorizontalAlignment::Left,
      sdl::graphic::LabelWidget::VerticalAlignment::Center,
      this,
      getBackgroundColor()
    );
    if (accuracyLabel == nullptr) {
      error(
        std::string("Could not create render settings"),
        std::string("Accuracy label not created")
      );
    }

    accuracyLabel->setMaxSize(szMax);

    sdl::graphic::TextBox* accuracyValue = new sdl::graphic::TextBox(
      getAccuracyTextBoxName(),
      getGeneralTextFont(),
      std::to_string(getDefaultAccuracy()),
      getGeneralTextSize(),this
    );
    if (accuracyValue == nullptr) {
      error(
        std::string("Could not create render settings"),
        std::string("Accuracy textbox not created")
      );
    }

    accuracyValue->setMaxSize(szMax);

    // Create iterations data.
    sdl::graphic::LabelWidget* maxIterLabel = new sdl::graphic::LabelWidget(
      getMaxIterationsLabelName(),
      std::string("Iterations:"),
      getGeneralTextFont(),
      getGeneralTextSize(),
      sdl::graphic::LabelWidget::HorizontalAlignment::Left,
      sdl::graphic::LabelWidget::VerticalAlignment::Center,
      this,
      getBackgroundColor()
    );
    if (maxIterLabel == nullptr) {
      error(
        std::string("Could not create render settings"),
        std::string("Max iterations label not created")
      );
    }

    maxIterLabel->setMaxSize(szMax);

    sdl::graphic::TextBox* maxIterValue = new sdl::graphic::TextBox(
      getMaxIterationsTextBoxName(),
      getGeneralTextFont(),
      std::to_string(getDefaultMaxIterations()),
      getGeneralTextSize(),this
    );
    if (powerValue == nullptr) {
      error(
        std::string("Could not create render settings"),
        std::string("Max iterations textbox not created")
      );
    }

    maxIterValue->setMaxSize(szMax);

    // Create apply button.
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
        std::string("Could not create render settings"),
        std::string("Apply button not created")
      );
    }

    powerLabel->allowLog(false);
    powerValue->allowLog(false);
    accuracyLabel->allowLog(false);
    accuracyValue->allowLog(false);
    maxIterLabel->allowLog(false);
    maxIterValue->allowLog(false);
    apply->allowLog(false);

    // Build layout.
    layout->addItem(powerLabel);
    layout->addItem(powerValue);
    layout->addItem(accuracyLabel);
    layout->addItem(accuracyValue);
    layout->addItem(maxIterLabel);
    layout->addItem(maxIterValue);
    layout->addItem(apply);

    // Connect the button `onClick` signal to the local slot in
    // order to be able to notify rendering properties.
    apply->onClick.connect_member<RenderSettings>(
      this,
      &RenderSettings::onApplyButtonClicked
    );
  }

  void
  RenderSettings::onApplyButtonClicked(const std::string& /*dummy*/) {
    // We need to retrieve the value for the power, the maximum iterations
    // and the accuracy.
    std::string powerStr, accStr, maxIterStr;

    sdl::graphic::TextBox* pTB = getPowerValueTextBox();
    sdl::graphic::TextBox* aTB = getAccuracyTextBox();
    sdl::graphic::TextBox* mTB = getMaxIterationsTextBox();

    if (pTB == nullptr) {
      log(
        std::string("Could not gather rendering properties (invalid power label)"),
        utils::Level::Error
      );

      return;
    }
    if (aTB == nullptr) {
      log(
        std::string("Could not gather rendering properties (invalid accuracy label)"),
        utils::Level::Error
      );

      return;
    }
    if (mTB == nullptr) {
      log(
        std::string("Could not gather rendering properties (invalid max iterations label)"),
        utils::Level::Error
      );

      return;
    }

    withSafetyNet(
      [&powerStr, &accStr, &maxIterStr, pTB, aTB, mTB]() {
        powerStr = pTB->getValue();
        accStr = aTB->getValue();
        maxIterStr = mTB->getValue();
      },
      std::string("RenderSettings::gatherProps")
    );

    // Convert each property.
    bool sP = false, sA = false, sM = false;
    float p = utils::convert(powerStr, getDefaultPower(), sP);
    float a = utils::convert(accStr, getDefaultAccuracy(), sA);
    float m = utils::convert(maxIterStr, getDefaultMaxIterations(), sM);

    if (!sP) {
      log(
        std::string("Could not convert provided power of \"") + powerStr + "\", using " + std::to_string(p) + " instead",
        utils::Level::Warning
      );
    }
    if (!sA) {
      log(
        std::string("Could not convert provided accuracy of \"") + accStr + "\", using " + std::to_string(a) + " instead",
        utils::Level::Warning
      );
    }
    if (!sM) {
      log(
        std::string("Could not convert provided max iterations count of \"") + maxIterStr + "\", using " + std::to_string(m) + " instead",
        utils::Level::Warning
      );
    }

    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Create the rendering properties object.
    // TODO: Implementation.

    // Notify listeners through the signal.
    onRenderingSettingsChanged.safeEmit(
      std::string("onRenderingSettingsChanged()")
    );
  }

}
