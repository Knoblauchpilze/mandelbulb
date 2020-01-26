
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

    // Create hit threshold data.
    sdl::graphic::LabelWidget* hitThreshLabel = new sdl::graphic::LabelWidget(
      getHitThresholdLabelName(),
      std::string("Hit threshold:"),
      getGeneralTextFont(),
      getGeneralTextSize(),
      sdl::graphic::LabelWidget::HorizontalAlignment::Left,
      sdl::graphic::LabelWidget::VerticalAlignment::Center,
      this,
      getBackgroundColor()
    );
    if (hitThreshLabel == nullptr) {
      error(
        std::string("Could not create render settings"),
        std::string("Hit threshold label not created")
      );
    }

    hitThreshLabel->setMaxSize(szMax);

    ss.str("");
    ss.clear();
    ss << std::setprecision(1) << getDefaultHitThreshold();

    sdl::graphic::TextBox* hitThreshValue = new sdl::graphic::TextBox(
      getHitThresholdTextBoxName(),
      getGeneralTextFont(),
      ss.str(),
      getGeneralTextSize(),this
    );
    if (hitThreshValue == nullptr) {
      error(
        std::string("Could not create render settings"),
        std::string("Hit threshol textbox not created")
      );
    }

    hitThreshValue->setMaxSize(szMax);

    // Create max ray steps data.
    sdl::graphic::LabelWidget* maxRayLabel = new sdl::graphic::LabelWidget(
      getRayStepsLabelName(),
      std::string("Max ray steps:"),
      getGeneralTextFont(),
      getGeneralTextSize(),
      sdl::graphic::LabelWidget::HorizontalAlignment::Left,
      sdl::graphic::LabelWidget::VerticalAlignment::Center,
      this,
      getBackgroundColor()
    );
    if (maxRayLabel == nullptr) {
      error(
        std::string("Could not create render settings"),
        std::string("Max ray steps label not created")
      );
    }

    maxRayLabel->setMaxSize(szMax);

    sdl::graphic::TextBox* maxRayValue = new sdl::graphic::TextBox(
      getRayStepsTextBoxName(),
      getGeneralTextFont(),
      std::to_string(getDefaultRaySteps()),
      getGeneralTextSize(),this
    );
    if (maxRayValue == nullptr) {
      error(
        std::string("Could not create render settings"),
        std::string("Max ray steps textbox not created")
      );
    }

    maxRayValue->setMaxSize(szMax);

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
    hitThreshLabel->allowLog(false);
    hitThreshValue->allowLog(false);
    maxRayLabel->allowLog(false);
    maxRayValue->allowLog(false);
    apply->allowLog(false);

    // Build layout.
    layout->addItem(powerLabel);
    layout->addItem(powerValue);
    layout->addItem(accuracyLabel);
    layout->addItem(accuracyValue);
    layout->addItem(hitThreshLabel);
    layout->addItem(hitThreshValue);
    layout->addItem(maxRayLabel);
    layout->addItem(maxRayValue);
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
    std::string powerStr, accStr, hitThreshStr, rayStStr;

    sdl::graphic::TextBox* powerTB = getPowerValueTextBox();
    sdl::graphic::TextBox* accTB = getAccuracyTextBox();
    sdl::graphic::TextBox* hitTB = getHitThresholdTextBox();
    sdl::graphic::TextBox* rayStTB = getRayStepsTextBox();

    if (powerTB == nullptr) {
      log(
        std::string("Could not gather rendering properties (invalid power label)"),
        utils::Level::Error
      );

      return;
    }
    if (accTB == nullptr) {
      log(
        std::string("Could not gather rendering properties (invalid accuracy label)"),
        utils::Level::Error
      );

      return;
    }
    if (hitTB == nullptr) {
      log(
        std::string("Could not gather rendering properties (invalid hit threshold label)"),
        utils::Level::Error
      );

      return;
    }
    if (rayStTB == nullptr) {
      log(
        std::string("Could not gather rendering properties (invalid ray steps label)"),
        utils::Level::Error
      );

      return;
    }

    withSafetyNet(
      [&powerStr, &accStr, powerTB, accTB]() {
        powerStr = powerTB->getValue();
        accStr = accTB->getValue();
      },
      std::string("RenderSettings::gatherProps")
    );

    withSafetyNet(
      [&hitThreshStr, &rayStStr, hitTB, rayStTB]() {
        hitThreshStr = hitTB->getValue();
        rayStStr = rayStTB->getValue();
      },
      std::string("RenderSettings::gatherProps")
    );

    // Convert each property.
    bool sPower = false, sAcc = false, sHitThr = false, sRaySt = false;

    float power = utils::convert(powerStr, getDefaultPower(), sPower);
    unsigned acc = utils::convert(accStr, getDefaultAccuracy(), sAcc);
    float hitThresh = utils::convert(hitThreshStr, getDefaultHitThreshold(), sHitThr);
    unsigned raySt = utils::convert(rayStStr, getDefaultRaySteps(), sRaySt);

    if (!sPower) {
      log(
        std::string("Could not convert provided power of \"") + powerStr + "\", using " + std::to_string(power) + " instead",
        utils::Level::Warning
      );
    }
    if (!sAcc) {
      log(
        std::string("Could not convert provided accuracy of \"") + accStr + "\", using " + std::to_string(acc) + " instead",
        utils::Level::Warning
      );
    }
    if (!sHitThr) {
      log(
        std::string("Could not convert provided hit threshold of \"") + hitThreshStr + "\", using " + std::to_string(hitThresh) + " instead",
        utils::Level::Warning
      );
    }
    if (!sRaySt) {
      log(
        std::string("Could not convert provided max ray steps of \"") + rayStStr + "\", using " + std::to_string(raySt) + " instead",
        utils::Level::Warning
      );
    }

    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Create the rendering properties object.
    RenderProperties props = RenderProperties{
      acc,
      power,
      hitThresh,
      raySt,
      getDefaultBailout()
    };

    // Notify listeners through the signal.
    onRenderingSettingsChanged.safeEmit(
      std::string("onRenderingSettingsChanged()"),
      props
    );
  }

}
