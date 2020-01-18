#ifndef    RENDER_MENU_HH
# define   RENDER_MENU_HH

# include <maths_utils/Size.hh>
# include <sdl_core/SdlWidget.hh>
# include <sdl_graphic/ProgressBar.hh>
# include <sdl_graphic/LabelWidget.hh>

namespace mandelbulb {

  class RenderMenu: public sdl::core::SdlWidget {
    public:

      RenderMenu(const utils::Sizef& sizeHint = utils::Sizef(),
                 sdl::core::SdlWidget* parent = nullptr);

      virtual ~RenderMenu();

      void
      onCompletionChanged(float perc);

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
      float
      getPercentageTextMaxWidth() noexcept;

      static
      const char*
      getProgressBarName() noexcept;

      sdl::graphic::ProgressBar*
      getProgressBar();

      static
      const char*
      getPercentageLabelName() noexcept;

      sdl::graphic::LabelWidget*
      getPercentageLabel();

      void
      build();

    private:
  };

}

# include "RenderMenu.hxx"

#endif    /* RENDER_MENU_HH */
