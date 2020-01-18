#ifndef    LIGHT_SETTINGS_HH
# define   LIGHT_SETTINGS_HH

# include <mutex>
# include <vector>
# include <sdl_engine/Color.hh>
# include <maths_utils/Size.hh>
# include <sdl_core/SdlWidget.hh>
# include <sdl_graphic/Button.hh>
# include <sdl_graphic/LabelWidget.hh>
# include <sdl_graphic/TextBox.hh>
# include <sdl_graphic/SelectorWidget.hh>

namespace mandelbulb {

  class LightSettings: public sdl::core::SdlWidget {
    public:

      LightSettings(const utils::Sizef& sizeHint = utils::Sizef(),
                    sdl::core::SdlWidget* parent = nullptr);

      virtual ~LightSettings();

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
      unsigned
      getLightsCount() noexcept;

      static
      float
      getLabelMaxWidth() noexcept;

      static
      std::string
      generateNameForLightToggle(unsigned id) noexcept;

      static
      float
      getLightToggleButtonBordersSize() noexcept;

      sdl::graphic::Button*
      getButtonForLight(unsigned id);

      static
      float
      getDefaultLightPosition(unsigned id,
                              char axis) noexcept;

      static
      std::string
      generateNameForLightPositionLabel(unsigned id,
                                        char axis) noexcept;

      static
      std::string
      generateNameForLightPositionValue(unsigned id,
                                        char axis) noexcept;

      sdl::graphic::TextBox*
      getTextBoxForLightPosition(unsigned id,
                                 char axis);

      static
      float
      getDefaultLightPower() noexcept;

      static
      std::string
      generateNameForLightPowerLabel(unsigned id) noexcept;

      static
      std::string
      generateNameForLightPowerValue(unsigned id) noexcept;

      sdl::graphic::TextBox*
      getTextBoxForLightPower(unsigned id);

      static
      std::string
      generateNameForLightColorLabel(unsigned id) noexcept;

      static
      std::string
      generateNameForLightColorValue(unsigned id) noexcept;

      /**
       * @brief - Used to create the complete palette object used to represent
       *          a color to assign to a light. This include populating the item
       *          with relevant colors.
       *          The colors are taken from the internal map and the `id-th`
       *          color is used as the active one.
       * @param id - the index of the palette to create.
       * @return - the object representing the palette.
       */
      sdl::graphic::SelectorWidget*
      createPaletteFromIndex(unsigned id);

      sdl::graphic::SelectorWidget*
      getPaletteForLightColor(unsigned id);

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

      /**
       * @brief - A mutex to protect the internal properties of this widget.
       */
      mutable std::mutex m_propsLocker;

      /**
       * @brief - An ordered list of colors which is assigned to the item
       *          allowing to select colors for a light. This ordered list
       *          is used when the rendering options should be packaged to
       *          make the link between the active element in a single
       *          palette and the color that is expected to be added to the
       *          props.
       */
      std::vector<sdl::core::engine::Color> m_colors;
  };

}

# include "LightSettings.hxx"

#endif    /* LIGHT_SETTINGS_HH */
