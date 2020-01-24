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

      ~MandelbulbRenderer();

    protected:

      /**
       * @brief - Reimplementation of the base class method to provide update of the
       *          camera used to visualize the fractal object when the size of this
       *          component is changed.
       * @param window - the available size to perform the update.
       */
      void
      updatePrivate(const utils::Boxf& window) override;

      /**
       * @brief - Reimplementation of the base class method to handle the repaint of
       *          the texture representing the fractal and its display.
       */
      void
      drawContentPrivate(const utils::Uuid& uuid,
                         const utils::Boxf& area) override;

      /**
       * @brief - Specialziation of the parent method in order to perform the
       *          scrolling on this object. It actually accounts for rotating
       *          the camera to change the viewpoint on the fractal object.
       *          How the camera is actually rotated is not specified at this
       *          level.
       *          The interface is similar to what is expected by the parent
       *          class (see `ScrollableWidget` for more details).
       * @param posToFix - the position in local coordinate frame corresponding
       *                   to the position that should be fixed during the
       *                   scroll operation.
       * @param whereTo - the new position of the `posToFix`. Corresponds to
       *                  the `posToFix` where the `motion` has been applied.
       * @param motion - the motion to apply.
       * @param notify - indication as to this method should emit some signals
       *                 like `onHorizontalAxisChanged`.
       * @return - `true` if the actual rendering area has been changed, and
       *           `false` otherwise.
       */
      bool
      handleContentScrolling(const utils::Vector2f& posToFix,
                             const utils::Vector2f& whereTo,
                             const utils::Vector2f& motion,
                             bool notify = true) override;

      /**
       * @brief - Reimplementation of the base class method to detect whenever the
       *          arrow keys are pressed in order to rotate/change the camera.
       * @param e - the event to be interpreted.
       * @return - `true` if the event was recognized and `false` otherwise.
       */
      bool
      keyPressEvent(const sdl::core::engine::KeyEvent& e) override;

      /**
       * @brief - Reimplementation of the base class method to detect whenever the
       *          mouse moves inside the widget. This allows to provide notification
       *          to external listeners by converting the position into a real world
       *          coordinates.
       * @param e - the event to be interpreted.
       * @return - `true` if the event was recognized and `false` otherwise.
       */
      bool
      mouseMoveEvent(const sdl::core::engine::MouseEvent& e) override;

      /**
       * @brief - Reimplementation of the base class method to detect when the wheel
       *          is used: this should trigger a motion of the camera along the view
       *          vector to get closer/farther from the fractal object.
       * @param e - the event to be interpreted.
       * @return - `true` if the event was recognized and `false` otherwise.
       */
      bool
      mouseWheelEvent(const sdl::core::engine::MouseEvent& e) override;

    private:

      /**
       * @brief - Used to retrieve the default factor to use when zooming in.
       * @return - a factor suitable for zooming in operations.
       */
      static
      constexpr float
      getDefaultZoomInFactor() noexcept;

      /**
       * @brief - Used to retrieve the default factor to use when zooming out.
       * @return - a factor suitable for zooming out operations.
       */
      static
      constexpr float
      getDefaultZoomOutFactor() noexcept;

      /**
       *  @brief - Default value that can be used to rotate the camera about an
       *           axis.
       * @return - an angle in radians that is suited for small step rotations
       *           about an axis.
       */
      static
      constexpr float
      getArrowKeyRotationAngle() noexcept;

      /**
       * @brief - Return a minimum viewing distance. When using the mouse wheel
       *          the zooming in behavior nothing will happen if the viewing
       *          distance is already smaller than this value. This avoid getting
       *          too close of the fractal.
       * @return - the minimum viewing distance.
       */
      static
      constexpr float
      getMinimumViewingDistance() noexcept;

      /**
       * @brief - Similar to `getMinimumViewingDistance` but provide a maximum
       *          viewing distance above which the fractal does not show any
       *          interesting details.
       * @return - the maximum viewing distance.
       */
      static
      constexpr float
      getMaximumViewingDistance() noexcept;

      /**
       * @brief - Provide a default value for the number of radians represented
       *          by a single pixel. This is used in case of scrolling where we
       *          want to transform a motion of the mouse in terms of real world
       *          coordinates.
       *          Larger values will make the scrolling quicker but also somewhat
       *          less accurate.
       * @return - a ratio indicating how many radians are represented by a single
       *           pixel of the screen.
       */
      static
      constexpr float
      getPixelToRadiansRatio() noexcept;

      /**
       * @brief - The palette to use to render the fractal's data. Provide a nice
       *          set of colors to use to color points based on their distance.
       * @return - a valid gradient to use to represent the fractal's data.
       */
      static
      sdl::core::engine::GradientShPtr
      generateDefaultPalette() noexcept;

      /**
       * @brief - The default range to apply to the palette's gradient. Defines
       *          how often the colors will be repeated and over which distance.
       * @return - a suitable distance to wrap colors.
       */
      static
      float
      getDefaultPaletteRange() noexcept;

      /**
       * @brief - Used to provide a default color to use for pixels where no fractal
       *          data is to be found. This will represent the background color used
       *          to render the fractal.
       * @return - a color suited to display the fractal.
       */
      static
      sdl::core::engine::Color
      getNoDataColor() noexcept;

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

      /**
       * @brief - Used to clear the texture associated to this fractal.
       */
      void
      clearTiles();

      /**
       * @brief - Used to determine whether the tiles have changed since the creation
       *          of the texture representing them. If this is the case it means that
       *          the `m_tex` should be recreated.
       *          Assumes that the locker is already acquired.
       */
      bool
      tilesChanged() const noexcept;

      /**
       * @brief - Used to specify that the tiles have changed and thus that the `m_tex`
       *          texture should be recreated on the next call to `drawContentPrivate`.
       *          Assumes that the locker is already acquired.
       */
      void
      setTilesChanged() noexcept;

      /**
       * @brief - Performs the creation of the texture representing this fractal from
       *          the data associated to it. Assumes that the locker is already acquired.
       */
      void
      loadTiles();

      /**
       * @brief - Used to notify external listeners about a change in the pointed coords
       *          by the mouse. This can eithe result in a motion of the mouse or a move
       *          in the position/rotation of the camera.
       * @param coord - the mouse coordinates in global coordinate frame.
       */
      void
      notifyCoordinatesChanged(const utils::Vector2f& coord) noexcept;

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

      /**
       * @brief - The index returned by the engine for the texture representing the
       *          fractal on screen. Its size is consistent with the size of the cam
       *          defined by the `m_fractal` object and is updated whenever some tiles
       *          are received and marked ready for display.
       *          The texture is used as long as possible that is until recomputing
       *          parts of it make it invalid. As long as the `m_tilesRendered` area
       *          is `false` the texture can be used as is.
       */
      utils::Uuid m_tex;

      /**
       * @brief - This value indicates whether the `m_tex` identifier is still valid
       *          or not. Each time a tile is rendered this value is set to `true` to
       *          indicate that the texture representing the fractal needs to be
       *          updated.
       */
      bool m_tilesRendered;

      /**
       * @brief - The rendering palette of the fractal. This is used when the fractal's
       *          data is interpreted to render it with some nice colors.
       */
      RenderPalette m_palette;

    public:

      /**
       * @brief - Signal notifying external listeners that the coordinates of the
       *          mouse in real world's coordinate frame have changed.
       */
      utils::Signal<const utils::Vector3f&> onCoordinatesChanged;

      /**
       * @brief - Signal notifying external listeners that the depth of the point
       *          under the mouse's cursor has changed depth.
       */
      utils::Signal<float> onDepthChanged;
  };

}

# include "MandelbulbRenderer.hxx"

#endif    /* MANDELBULB_RENDERER_HH */
