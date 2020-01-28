#ifndef    FRACTAL_SETTINGS_HH
# define   FRACTAL_SETTINGS_HH

# include <mutex>
# include <maths_utils/Size.hh>
# include <sdl_core/SdlWidget.hh>
# include <sdl_graphic/SelectorWidget.hh>
# include <sdl_graphic/TextBox.hh>
# include <core_utils/Signal.hh>
# include "RenderProperties.hh"

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
      float
      getLabelMaxHeight() noexcept;

      sdl::graphic::SelectorWidget*
      createPalette(const std::string& name,
                    unsigned index);

      // Fractal color.
      static
      const char*
      getFractalColorLabelName() noexcept;

      static
      const char*
      getFractalColorPaletteName() noexcept;

      static
      sdl::core::engine::Color
      getDefaultFractalColor() noexcept;

      sdl::graphic::SelectorWidget*
      getFractalColorPalette();

      // Fractal color blend.
      static
      const char*
      getFractalColorBlendingLabelName() noexcept;

      static
      const char*
      getFractalColorBlendingValueName() noexcept;

      static
      float
      getDefaultFractalColorBlending();

      sdl::graphic::TextBox*
      getFractalColorBlendingValue();

      // No data color.
      static
      const char*
      getNoDataColorLabelName() noexcept;

      static
      const char*
      getNoDataColorPaletteName() noexcept;

      sdl::core::engine::Color
      getDefaultNoDataColor() noexcept;

      sdl::graphic::SelectorWidget*
      getNoDataColorPalette();

      // Tonemapping exposure.
      static
      const char*
      getExposureLabelName() noexcept;

      static
      const char*
      getExposureValueName() noexcept;

      static
      float
      getDefaultExposureValue() noexcept;

      sdl::graphic::TextBox*
      getExposureValue();

      // Tonemapping burnout.
      static
      const char*
      getBurnoutLabelName() noexcept;

      static
      const char*
      getBurnoutValueName() noexcept;

      static
      float
      getDefaultBurnoutValue() noexcept;

      sdl::graphic::TextBox*
      getBurnoutValue();

      static
      const char*
      getApplyButtonName() noexcept;

      static
      float
      getApplyButtonBordersSize() noexcept;

      void
      build();

      void
      generatePalette() noexcept;

      void
      onApplyButtonClicked(const std::string& dummy);

    private:

      std::mutex m_propsLocker;

      /**
       * @brief - An ordered list of colors which is assigned to the item
       *          allowing to select colors for the fractal. This ordered
       *          list is used when the shading options should be packaged
       *          to make the link between the active element in a single
       *          palette and the color that is expected to be added to the
       *          props.
       */
      std::vector<sdl::core::engine::Color> m_colors;

    public:

      utils::Signal<ShadingProperties> onShadingPropertiesChanged;
  };

}

# include "FractalSettings.hxx"

#endif    /* FRACTAL_SETTINGS_HH */
