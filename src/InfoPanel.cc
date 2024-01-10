
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
  InfoPanel::onCoordinatesChanged(const utils::Vector3f& coords) {
    // Retrieve the coordinate label and update its associated text.
    sdl::graphic::LabelWidget* label = getCoordLabel();

    if (label == nullptr) {
      warn("Could not update coordinates to " + coords.toString() + ", unable to fetch display label");
      return;
    }

    std::stringstream ss;
    std::string xStr("-"), yStr("-"), zStr("-");

    // Consider that minimum value is an invalid one.
    if (coords.x() != std::numeric_limits<float>::lowest()) {
      ss << std::setprecision(4) << coords.x();
      xStr = ss.str();

      ss.clear();
      ss.str("");
    }

    if (coords.y() != std::numeric_limits<float>::lowest()) {
      ss << std::setprecision(4) << coords.y();
      yStr = ss.str();

      ss.clear();
      ss.str("");
    }

    if (coords.z() != std::numeric_limits<float>::lowest()) {
      ss << std::setprecision(4) << coords.z();
      zStr = ss.str();
    }

    label->setText("x: " + xStr + ", y: " + yStr + ", z: " + zStr);
  }

  void
  InfoPanel::onDepthChanged(float depth) {
    // Retrieve the depth label and update its associated text.
    sdl::graphic::LabelWidget* label = getDepthLabel();

    if (label == nullptr) {
      warn("Could not update depth to " + std::to_string(depth) + ", unable to fetch display label");
      return;
    }

    std::string d("-");

    if (depth >= 0.0f) {
      std::stringstream ss;
      ss << std::setprecision(4) << depth;

      d = ss.str();
    }

    label->setText("Depth: " + d);
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
      std::string("x: -, y: -, z: -"),
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

    // Build layout.
    layout->addItem(coords);
    layout->addItem(depth);
  }

}
