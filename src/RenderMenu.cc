
# include "RenderMenu.hh"
# include <sstream>
# include <iomanip>
# include <sdl_graphic/LinearLayout.hh>

namespace mandelbulb {

  RenderMenu::RenderMenu(const utils::Sizef& sizeHint,
                         sdl::core::SdlWidget* parent):
    sdl::core::SdlWidget(std::string("render_menu"),
                         sizeHint,
                         parent,
                         getBackgroundColor())
  {
    build();
  }

  void
  RenderMenu::onCompletionChanged(float perc) {
    // Update both the progress bar and the label.
    sdl::graphic::ProgressBar* bar = getProgressBar();

    if (bar != nullptr) {
      bar->setCompletion(perc);
    }
    else {
      log(
        std::string("Could not update completion to ") + std::to_string(perc) + ", unable to fetch progress bar",
        utils::Level::Error
      );
    }

    sdl::graphic::LabelWidget* label = getPercentageLabel();
    if (label != nullptr) {
      std::stringstream ss;
      ss << std::setprecision(4) << (100.0f * perc);

      label->setText(ss.str() + "%");
    }
    else {
      log(
        std::string("Could not update completion to ") + std::to_string(perc) + ", unable to fetch display label",
        utils::Level::Error
      );
    }
  }

  void
  RenderMenu::build() {
    // This component uses a horizotnal linear layout to position the
    // internal components.
    sdl::graphic::LinearLayoutShPtr layout = std::make_shared<sdl::graphic::LinearLayout>(
      "render_menu_layout",
      this,
      sdl::graphic::LinearLayout::Direction::Horizontal,
      getGlobalMargins(),
      getComponentMargins()
    );

    setLayout(layout);

    // Create the progress bar data.
    sdl::graphic::ProgressBar* bar = new sdl::graphic::ProgressBar(
      getProgressBarName(),
      this
    );
    if (bar == nullptr) {
      error(
        std::string("Could not create render menu"),
        std::string("Progress bar not created")
      );
    }

    // Build percentage label data.
    sdl::graphic::LabelWidget* label = new sdl::graphic::LabelWidget(
      getPercentageLabelName(),
      std::string("0%"),
      getGeneralTextFont(),
      getGeneralTextSize(),
      sdl::graphic::LabelWidget::HorizontalAlignment::Center,
      sdl::graphic::LabelWidget::VerticalAlignment::Center,
      this,
      getBackgroundColor()
    );
    if (label == nullptr) {
      error(
        std::string("Could not create render menu"),
        std::string("Percentage label not created")
      );
    }

    label->setMaxSize(
      utils::Sizef(
        getPercentageTextMaxWidth(),
        std::numeric_limits<float>::max()
      )
    );

    // Build layout.
    layout->addItem(bar);
    layout->addItem(label);
  }

}
