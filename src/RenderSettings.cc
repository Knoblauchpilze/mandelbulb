
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
    if (maxIterValue == nullptr) {
      error(
        std::string("Could not create render settings"),
        std::string("Max iterations textbox not created")
      );
    }

    maxIterValue->setMaxSize(szMax);

    // Create bailout data.
    sdl::graphic::LabelWidget* bailoutLabel = new sdl::graphic::LabelWidget(
      getBailoutLabelName(),
      std::string("Bailout:"),
      getGeneralTextFont(),
      getGeneralTextSize(),
      sdl::graphic::LabelWidget::HorizontalAlignment::Left,
      sdl::graphic::LabelWidget::VerticalAlignment::Center,
      this,
      getBackgroundColor()
    );
    if (bailoutLabel == nullptr) {
      error(
        std::string("Could not create render settings"),
        std::string("Bailout label not created")
      );
    }

    bailoutLabel->setMaxSize(szMax);

    ss.str("");
    ss.clear();
    ss << std::setprecision(1) << getDefaultBailout();

    sdl::graphic::TextBox* bailoutValue = new sdl::graphic::TextBox(
      getBailoutTextBoxName(),
      getGeneralTextFont(),
      ss.str(),
      getGeneralTextSize(),this
    );
    if (bailoutValue == nullptr) {
      error(
        std::string("Could not create render settings"),
        std::string("Bailout textbox not created")
      );
    }

    bailoutValue->setMaxSize(szMax);

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
    maxIterLabel->allowLog(false);
    maxIterValue->allowLog(false);
    bailoutLabel->allowLog(false);
    bailoutValue->allowLog(false);
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
    layout->addItem(maxIterLabel);
    layout->addItem(maxIterValue);
    layout->addItem(bailoutLabel);
    layout->addItem(bailoutValue);
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
    std::string powerStr, accStr, maxIterStr, hitThreshStr, rayStStr, bailoutStr;

    sdl::graphic::TextBox* powerTB = getPowerValueTextBox();
    sdl::graphic::TextBox* accTB = getAccuracyTextBox();
    sdl::graphic::TextBox* maxItTB = getMaxIterationsTextBox();

    sdl::graphic::TextBox* hitTB = getHitThresholdTextBox();
    sdl::graphic::TextBox* rayStTB = getRayStepsTextBox();
    sdl::graphic::TextBox* bailoutTB = getBailoutTextBox();

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
    if (maxItTB == nullptr) {
      log(
        std::string("Could not gather rendering properties (invalid max iterations label)"),
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
    if (bailoutTB == nullptr) {
      log(
        std::string("Could not gather rendering properties (invalid bailout label)"),
        utils::Level::Error
      );

      return;
    }

    withSafetyNet(
      [&powerStr, &accStr, &maxIterStr, powerTB, accTB, maxItTB]() {
        powerStr = powerTB->getValue();
        accStr = accTB->getValue();
        maxIterStr = maxItTB->getValue();
      },
      std::string("RenderSettings::gatherProps")
    );

    withSafetyNet(
      [&hitThreshStr, &rayStStr, &bailoutStr, hitTB, rayStTB, bailoutTB]() {
        hitThreshStr = hitTB->getValue();
        rayStStr = rayStTB->getValue();
        bailoutStr = bailoutTB->getValue();
      },
      std::string("RenderSettings::gatherProps")
    );

    // Convert each property.
    bool sPower = false, sAcc = false, sMaxIt = false, sHitThr = false, sRaySt = false, sBailout = false;

    float power = utils::convert(powerStr, getDefaultPower(), sPower);
    unsigned acc = utils::convert(accStr, getDefaultAccuracy(), sAcc);
    unsigned maxIter = utils::convert(maxIterStr, getDefaultMaxIterations(), sMaxIt);
    float hitThresh = utils::convert(hitThreshStr, getDefaultHitThreshold(), sHitThr);
    unsigned raySt = utils::convert(rayStStr, getDefaultRaySteps(), sRaySt);
    float bailout = utils::convert(bailoutStr, getDefaultBailout(), sBailout);

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
    if (!sMaxIt) {
      log(
        std::string("Could not convert provided max iterations count of \"") + maxIterStr + "\", using " + std::to_string(maxIter) + " instead",
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
    if (!sBailout) {
      log(
        std::string("Could not convert provided bailout value of \"") + bailoutStr + "\", using " + std::to_string(bailout) + " instead",
        utils::Level::Warning
      );
    }

    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Create the rendering properties object.
    RenderProperties props = RenderProperties{
      maxIter,
      acc,
      power,
      hitThresh,
      raySt,
      bailout
    };

    // Notify listeners through the signal.
    onRenderingSettingsChanged.safeEmit(
      std::string("onRenderingSettingsChanged()"),
      props
    );
  }

}
