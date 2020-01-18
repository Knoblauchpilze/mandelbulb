
# include "InfoPanel.hh"
# include <sdl_graphic/LinearLayout.hh>
# include <iomanip>
# include <sstream>

namespace mandelbulb {

  InfoPanel::InfoPanel(const utils::Sizef& sizeHint,
                       sdl::core::SdlWidget* parent):
    sdl::core::SdlWidget(std::string("info_panel"),
                         sizeHint,
                         parent,
                         getBackgroundColor())
  {
    build();
  }

  void
  InfoPanel::onCoordinatesChanged(const utils::Vector2f& coords) {
    // Retrieve the coordinate label and update its associated text.
    sdl::graphic::LabelWidget* label = getCoordLabel();

    if (label == nullptr) {
      log(
        std::string("Could not update coordinates to ") + coords.toString() + ", unable to fetch display label",
        utils::Level::Error
      );

      return;
    }

    std::stringstream ss;

    ss << std::setprecision(4) << coords.x();
    std::string xStr = ss.str();

    ss.clear();
    ss.str("");

    ss << std::setprecision(4) << coords.y();
    std::string yStr = ss.str();

    label->setText("x: " + xStr + ", y: " + yStr);
  }

  void
  InfoPanel::onDepthChanged(float depth) {
    // Retrieve the depth label and update its associated text.
    sdl::graphic::LabelWidget* label = getDepthLabel();

    if (label == nullptr) {
      log(
        std::string("Could not update depth to ") + std::to_string(depth) + ", unable to fetch display label",
        utils::Level::Error
      );

      return;
    }

    std::stringstream ss;
    ss << std::setprecision(4) << depth;

    label->setText("Depth: " + ss.str());
  }

  void
  InfoPanel::onCameraChanged() {
    // TODO: Implementation.
  }

  void
  InfoPanel::build() {
    // This component uses a horizontal linear layout to position the
    // internal components.
    sdl::graphic::LinearLayoutShPtr layout = std::make_shared<sdl::graphic::LinearLayout>(
      "info_panel_layout",
      this,
      sdl::graphic::LinearLayout::Direction::Horizontal,
      getGlobalMargins(),
      getComponentMargins()
    );

    setLayout(layout);

    // Create coordinates label.
    sdl::graphic::LabelWidget* coords = new sdl::graphic::LabelWidget(
      getCoordLabelName(),
      std::string("x: -, y: -"),
      getGeneralTextFont(),
      getGeneralTextSize(),
      sdl::graphic::LabelWidget::HorizontalAlignment::Left,
      sdl::graphic::LabelWidget::VerticalAlignment::Center,
      this,
      getBackgroundColor()
    );
    if (coords == nullptr) {
      error(
        std::string("Could not create info panel"),
        std::string("Coordinate label not created")
      );
    }

    // Create depth label.
    sdl::graphic::LabelWidget* depth = new sdl::graphic::LabelWidget(
      getDepthLabelName(),
      std::string("Depth: -"),
      getGeneralTextFont(),
      getGeneralTextSize(),
      sdl::graphic::LabelWidget::HorizontalAlignment::Left,
      sdl::graphic::LabelWidget::VerticalAlignment::Center,
      this,
      getBackgroundColor()
    );
    if (depth == nullptr) {
      error(
        std::string("Could not create info panel"),
        std::string("Depth label not created")
      );
    }

    depth->setMaxSize(
      utils::Sizef(
        getDepthMaxWidth(),
        std::numeric_limits<float>::max()
      )
    );

    // Create camera position label.
    sdl::graphic::LabelWidget* cam = new sdl::graphic::LabelWidget(
      getCameraLabelName(),
      std::string("Cam: [x: -, y: -, z: -]"),
      getGeneralTextFont(),
      getGeneralTextSize(),
      sdl::graphic::LabelWidget::HorizontalAlignment::Left,
      sdl::graphic::LabelWidget::VerticalAlignment::Center,
      this,
      getBackgroundColor()
    );
    if (cam == nullptr) {
      error(
        std::string("Could not create info panel"),
        std::string("Camera label not created")
      );
    }

    // Build layout.
    layout->addItem(coords);
    layout->addItem(depth);
    layout->addItem(cam);
  }

}
