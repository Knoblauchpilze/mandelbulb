#ifndef    RENDER_MENU_HH
# define   RENDER_MENU_HH

# include <maths_utils/Size.hh>
# include <sdl_core/SdlWidget.hh>

namespace mandelbulb {

  class RenderMenu: public sdl::core::SdlWidget {
    public:

      RenderMenu(const utils::Sizef& sizeHint = utils::Sizef(),
                 sdl::core::SdlWidget* parent = nullptr);

      virtual ~RenderMenu();

    private:

      void
      build();

    private:
  };

}

# include "RenderMenu.hxx"

#endif    /* RENDER_MENU_HH */
