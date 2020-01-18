#ifndef    LIGHT_SETTINGS_HH
# define   LIGHT_SETTINGS_HH

# include <maths_utils/Size.hh>
# include <sdl_core/SdlWidget.hh>

namespace mandelbulb {

  class LightSettings: public sdl::core::SdlWidget {
    public:

      LightSettings(const utils::Sizef& sizeHint = utils::Sizef(),
                    sdl::core::SdlWidget* parent = nullptr);

      virtual ~LightSettings();

    private:

      void
      build();

    private:
  };

}

# include "LightSettings.hxx"

#endif    /* LIGHT_SETTINGS_HH */
