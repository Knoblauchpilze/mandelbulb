#ifndef    INFO_PANEL_HH
# define   INFO_PANEL_HH

# include <maths_utils/Size.hh>
# include <sdl_core/SdlWidget.hh>
# include <sdl_graphic/LabelWidget.hh>
# include <sdl_engine/Color.hh>
# include <maths_utils/Vector3.hh>
# include "Camera.hh"

namespace mandelbulb {

  class InfoPanel: public sdl::core::SdlWidget {
    public:

      InfoPanel(const utils::Sizef& sizeHint = utils::Sizef(),
                sdl::core::SdlWidget* parent = nullptr);

      virtual ~InfoPanel();

      void
      onCoordinatesChanged(const utils::Vector3f& coords);

      void
      onDepthChanged(float depth);

    private:

      static
      float
      getGlobalMargins() noexcept;

      static
      float
      getComponentMargins() noexcept;

      static
      const char*
      getGeneralTextFont() noexcept;

      static
      unsigned
      getGeneralTextSize() noexcept;

      static
      sdl::core::engine::Color
      getBackgroundColor() noexcept;

      static
      const char*
      getCoordLabelName();

      sdl::graphic::LabelWidget*
      getCoordLabel();

      static
      const char*
      getDepthLabelName();

      static
      float
      getDepthMaxWidth() noexcept;

      sdl::graphic::LabelWidget*
      getDepthLabel();

      void
      build();

    private:
  };

}

# include "InfoPanel.hxx"

#endif    /* INFO_PANEL_HH */
