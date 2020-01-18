#ifndef    RENDER_SETTINGS_HH
# define   RENDER_SETTINGS_HH

# include <maths_utils/Size.hh>
# include <sdl_core/SdlWidget.hh>

namespace mandelbulb {

  class RenderSettings: public sdl::core::SdlWidget {
    public:

      RenderSettings(const utils::Sizef& sizeHint = utils::Sizef(),
                     sdl::core::SdlWidget* parent = nullptr);

      virtual ~RenderSettings();

    private:

      void
      build();

    private:
  };

}

# include "RenderSettings.hxx"

#endif    /* RENDER_SETTINGS_HH */
