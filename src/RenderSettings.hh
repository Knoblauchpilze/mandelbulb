#ifndef    RENDER_SETTINGS_HH
# define   RENDER_SETTINGS_HH

# include <mutex>
# include <maths_utils/Size.hh>
# include <sdl_core/SdlWidget.hh>
# include <sdl_graphic/LabelWidget.hh>
# include <sdl_graphic/TextBox.hh>
# include <core_utils/Signal.hh>

namespace mandelbulb {

  class RenderSettings: public sdl::core::SdlWidget {
    public:

      RenderSettings(const utils::Sizef& sizeHint = utils::Sizef(),
                     sdl::core::SdlWidget* parent = nullptr);

      virtual ~RenderSettings();

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
      getDisplayElementMaxHeight() noexcept;

      static
      const char*
      getPowerLabelName() noexcept;

      static
      const char*
      getPowerTextBoxName() noexcept;

      static
      float
      getDefaultPower() noexcept;

      sdl::graphic::TextBox*
      getPowerValueTextBox();

      static
      const char*
      getAccuracyLabelName() noexcept;

      static
      const char*
      getAccuracyTextBoxName() noexcept;

      static
      unsigned
      getDefaultAccuracy() noexcept;

      sdl::graphic::TextBox*
      getAccuracyTextBox();

      static
      const char*
      getMaxIterationsLabelName() noexcept;

      static
      const char*
      getMaxIterationsTextBoxName() noexcept;

      static
      unsigned
      getDefaultMaxIterations() noexcept;

      sdl::graphic::TextBox*
      getMaxIterationsTextBox();

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

    public:

      utils::Signal<> onRenderingSettingsChanged;
  };

}

# include "RenderSettings.hxx"

#endif    /* RENDER_SETTINGS_HH */
