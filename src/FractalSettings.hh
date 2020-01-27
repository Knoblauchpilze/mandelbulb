#ifndef    FRACTAL_SETTINGS_HH
# define   FRACTAL_SETTINGS_HH

# include <mutex>
# include <maths_utils/Size.hh>
# include <sdl_core/SdlWidget.hh>

namespace mandelbulb {

  class FractalSettings: public sdl::core::SdlWidget {
    public:

      FractalSettings(const utils::Sizef& sizeHint = utils::Sizef(),
                      sdl::core::SdlWidget* parent = nullptr);

      virtual ~FractalSettings();

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
      getApplyButtonName() noexcept;

      static
      float
      getApplyButtonBordersSize() noexcept;

      void
      build();

      void
      onApplyButtonClicked(const std::string& dummy);

    private:

      std::mutex m_propsLocker;
  };

}

# include "FractalSettings.hxx"

#endif    /* FRACTAL_SETTINGS_HH */
