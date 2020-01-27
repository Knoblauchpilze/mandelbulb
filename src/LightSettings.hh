#ifndef    LIGHT_SETTINGS_HH
# define   LIGHT_SETTINGS_HH

# include <vector>
# include <sdl_engine/Color.hh>
# include <maths_utils/Size.hh>
# include <sdl_core/SdlWidget.hh>
# include <sdl_graphic/Button.hh>
# include <sdl_graphic/LabelWidget.hh>
# include <sdl_graphic/TextBox.hh>
# include <sdl_graphic/SelectorWidget.hh>
# include <core_utils/Signal.hh>
# include "Light.hh"

namespace mandelbulb {

  class LightSettings: public sdl::core::SdlWidget {
    public:

      LightSettings(const utils::Sizef& sizeHint = utils::Sizef(),
                    sdl::core::SdlWidget* parent = nullptr);

      virtual ~LightSettings();

      /**
       * @brief - Used to generate default lights as they will be filled
       *          in the layout defined for this element. This is useful
       *          to transmit the knowledge of the default lights in some
       *          other place of the application.
       * @return - a vector describing the light as they will be defined
       *           in the controls of this component.
       */
      static
      std::vector<LightShPtr>
      generateDefaultLights() noexcept;

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
      utils::Vector3f
      getLightCircleNormal() noexcept;

      static
      float
      getDefaultLightPositionCircleRadius() noexcept;

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
      generateNameForLightPowerValue(unsigned id) noexcept;

      sdl::graphic::TextBox*
      getTextBoxForLightPower(unsigned id);

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
       * @brief - An ordered list of colors which is assigned to the item
       *          allowing to select colors for a light. This ordered list
       *          is used when the rendering options should be packaged to
       *          make the link between the active element in a single
       *          palette and the color that is expected to be added to the
       *          props.
       */
      std::vector<sdl::core::engine::Color> m_colors;

    public:

      /**
       * @brief - External signal allowing to notify listeners that the
       *          lights described by this component have been changed.
       */
      utils::Signal<const std::vector<LightShPtr>&> onLightsChanged;
  };

}

# include "LightSettings.hxx"

#endif    /* LIGHT_SETTINGS_HH */
