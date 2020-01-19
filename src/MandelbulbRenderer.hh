#ifndef    MANDELBULB_RENDERER_HH
# define   MANDELBULB_RENDERER_HH

# include <mutex>
# include <sdl_graphic/ScrollableWidget.hh>
# include <core_utils/Signal.hh>
# include <maths_utils/Vector2.hh>
# include "Fractal.hh"

namespace mandelbulb {

  class MandelbulbRenderer: public sdl::graphic::ScrollableWidget {
    public:

      /**
       * @brief - Create a widget allowing to display a visual of the provided
       *          fractal object.
       * @param fractal - the fractal object to display in this widget.
       * @param sizeHint - a size hint to apply to this widget.
       * @param parent - the  parent widget (if any) to this widget.
       */
      MandelbulbRenderer(FractalShPtr fractal,
                         const utils::Sizef& sizeHint = utils::Sizef(),
                         sdl::core::SdlWidget* parent = nullptr);

      ~MandelbulbRenderer() = default;

    protected:

      /**
       * @brief - Reimplementation of the base class method to provide update of the
       *          camera used to visualize the fractal object when the size of this
       *          component is changed.
       * @param window - the available size to perform the update.
       */
      void
      updatePrivate(const utils::Boxf& window) override;

    private:

      /**
       * @brief - Used to build the layout for this component. It is also used
       *          to connect the needed signals from the thread pool so that we
       *          can react to raytracing tiles being computed.
       */
      void
      build();

      /**
       * @brief - Local slot allowing to detect whenever the underlying fractal
       *          has computed some more rendering data so that we can update
       *          this widget and have an up-to-date visual representation.
       */
      void
      onTilesRendered();

    private:

      /**
       * @brief - A mutex allowing to protect this widget from concurrent accesses.
       */
      std::mutex m_propsLocker;

      /**
       * @brief - The fractal object rendered in this component. It automatically
       *          handles its rendering and notifies this widget so that it can
       *          update the relevant parts of the display.
       */
      FractalShPtr m_fractal;

    public:

      /**
       * @brief - Signal notifying external listeners that the cooridnates of the
       *          mouse in real world's coordinate frame have changed.
       */
      utils::Signal<const utils::Vector2f&> onCoordinatesChanged;

      /**
       * @brief - Signal notifying external listeners that the depth of the point
       *          under the mouse's cursor has changed depth.
       */
      utils::Signal<float> onDepthChanged;
  };

}

# include "MandelbulbRenderer.hxx"

#endif    /* MANDELBULB_RENDERER_HH */
