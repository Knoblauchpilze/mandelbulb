#ifndef    INFO_PANEL_HH
# define   INFO_PANEL_HH

# include <maths_utils/Size.hh>
# include <sdl_core/SdlWidget.hh>

namespace mandelbulb {

  class InfoPanel: public sdl::core::SdlWidget {
    public:

      InfoPanel(const utils::Sizef& sizeHint = utils::Sizef(),
                sdl::core::SdlWidget* parent = nullptr);

      virtual ~InfoPanel();

    private:

      void
      build();

    private:
  };

}

# include "InfoPanel.hxx"

#endif    /* INFO_PANEL_HH */
