#ifndef    MANDELBULB_RENDERER_HXX
# define   MANDELBULB_RENDERER_HXX

# include "MandelbulbRenderer.hh"

namespace mandelbulb {

  inline
  MandelbulbRenderer::~MandelbulbRenderer() {
    // Protect from concurrent accesses
    Guard guard(m_propsLocker);

    clearTiles();
  }

  inline
  void
  MandelbulbRenderer::updatePrivate(const utils::Boxf& window) {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Use the base handler.
    sdl::core::SdlWidget::updatePrivate(window);

    // Update the camera dimensions.
    m_fractal->setCameraDims(window.toSize());
  }

  inline
  bool
  MandelbulbRenderer::handleContentScrolling(const utils::Vector2f& /*posToFix*/,
                                             const utils::Vector2f& /*whereTo*/,
                                             const utils::Vector2f& motion,
                                             bool /*notify*/)
  {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Rotate about the `z` axis of an angle corresponding to the conversion
    // between the input motion and real world radians.
    m_fractal->rotateCamera(utils::Vector2f(motion.x() * getPixelToRadiansRatio(), 0.0f));

    // Notify the caller that we changed the area.
    return true;
  }

  inline
  bool
  MandelbulbRenderer::mouseMoveEvent(const sdl::core::engine::MouseEvent& e) {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    notifyCoordinatesChanged(e.getMousePosition());

    // Use the base handler to provide a return value.
    return sdl::graphic::ScrollableWidget::mouseMoveEvent(e);
  }

  inline
  bool
  MandelbulbRenderer::mouseWheelEvent(const sdl::core::engine::MouseEvent& e) {
    // We want to trigger zooming operations only when the mouse is inside
    // this widget.
    if (isMouseInside()) {
      // Protect from concurrent accesses to perform the zoom operation and
      // also schedule the rendering.
      utils::Vector2i motion = e.getScroll();
      bool zoomIn = motion.y() > 0;

      // Protect from concurrent accesses.
      Guard guard(m_propsLocker);

      // Retrieve an estimation of the distance of the camera to the fractal.
      float z = m_fractal->getDistance();
      float d = m_fractal->getDistanceEstimation();

      // Compute the new distance after the zooming in/out operation.
      float factor = zoomIn > 0 ? getDefaultZoomInFactor() : getDefaultZoomOutFactor();
      float newZ = (z - d) + d / factor;

      // Compute the new distance to the fractal.
      float newD = d - (z - newZ);

      log("DE: " + std::to_string(d) + ", d: " + std::to_string(z) + ", new: " + std::to_string(newZ) + ", newDE: " + std::to_string(newD) + ", min: " + std::to_string(getMinimumViewingDistance()));

      // Avoid zooming in if we're already really close from the fractal and zoom out
      // if we're already too far.
      if ((zoomIn && newD > getMinimumViewingDistance()) ||
          (!zoomIn && newD < getMaximumViewingDistance()))
      {
        // Perform the zoom in/out operation: we will zoom in half the distance
        // to the fractal and zoom out by doubling the current distance.
        // So for example if we're currently at `z` and the distance to reach the
        // the fractal is `d` then we will move to `z - d + d / zoomIn` which set
        // a distance to `zoomIn` closer to the fractal.
        m_fractal->updateDistance(newZ);
      }
    }

    return sdl::graphic::ScrollableWidget::mouseWheelEvent(e);;
  }

  inline
  constexpr float
  MandelbulbRenderer::getDefaultZoomInFactor() noexcept {
    return 2.0f;
  }

  inline
  constexpr float
  MandelbulbRenderer::getDefaultZoomOutFactor() noexcept {
    // Inverse of the default zoom in factor.
    return 1.0f / getDefaultZoomInFactor();
  }

  inline
  constexpr float
  MandelbulbRenderer::getArrowKeyRotationAngle() noexcept {
    return 0.314159f;
  }

  inline
  constexpr float
  MandelbulbRenderer::getMinimumViewingDistance() noexcept {
    return 0.01f;
  }

  inline
  constexpr float
  MandelbulbRenderer::getMaximumViewingDistance() noexcept {
    return 8.0f;
  }

  inline
  constexpr float
  MandelbulbRenderer::getPixelToRadiansRatio() noexcept {
    // Assume that we need 200 pixels to make a half-turn.
    return 3.1415926535f / 200.0f;
  }

  sdl::core::engine::GradientShPtr
  MandelbulbRenderer::generateDefaultPalette() noexcept {
    sdl::core::engine::GradientShPtr gr = std::make_shared<sdl::core::engine::Gradient>(
      std::string("fractal_palette"),
      sdl::core::engine::gradient::Mode::Linear
    );

    gr->setColorAt(0.0f, sdl::core::engine::Color::fromRGB(0.0f, 0.5f, 0.0f));
    gr->setColorAt(0.5f, sdl::core::engine::Color::fromRGB(0.0f, 1.0f, 0.0f));
    gr->makeWrap();

    return gr;
  }

  float
  MandelbulbRenderer::getDefaultPaletteRange() noexcept {
    return 10.0f;
  }

  inline
  sdl::core::engine::Color
  MandelbulbRenderer::getNoDataColor() noexcept {
    return sdl::core::engine::Color::NamedColor::Black;
  }

  inline
  void
  MandelbulbRenderer::onTilesRendered() {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Create and post a repaint event.
    postEvent(
      std::make_shared<sdl::core::engine::PaintEvent>(
        mapToGlobal(LayoutItem::getRenderingArea(), false)
      )
    );

    // Dirty the cache of the texture so that we actually repaint the
    // fractal upon receiving the paint event.
    setTilesChanged();
  }

  inline
  void
  MandelbulbRenderer::clearTiles() {
    if (m_tex.valid()) {
      getEngine().destroyTexture(m_tex);
      m_tex.invalidate();
    }
  }

  inline
  bool
  MandelbulbRenderer::tilesChanged() const noexcept {
    return m_tilesRendered;
  }

  inline
  void
  MandelbulbRenderer::setTilesChanged() noexcept {
    m_tilesRendered = true;
  }

  inline
  void
  MandelbulbRenderer::notifyCoordinatesChanged(const utils::Vector2f& coord) noexcept {
    // Note that we only want to notify coordinates in case the mouse is inside the
    // widget otherwise it makes no sense.
    if (!isMouseInside()) {
      return;
    }

    // Convert the coordinates to local coordinate frame.
    utils::Vector2f fConv = mapFromGlobal(coord);

    // Convert to integer camera plane coordinates.
    utils::Vector2i conv(
      static_cast<int>(std::round(fConv.x())),
      static_cast<int>(std::round(fConv.y()))
    );

    // Query the fractal object to get the point at these coordinates.
    bool hit = false;
    utils::Vector3f wCoords;

    float depth = m_fractal->getPoint(conv, wCoords, hit);

    // Notify external listeners.
    onCoordinatesChanged.safeEmit(
      std::string("onCoordinatesChanged(") + wCoords.toString() + ")",
      wCoords
    );

    onDepthChanged.safeEmit(
      std::string("onDepthChanged(") + std::to_string(depth) + ")",
      depth
    );
  }

}

#endif    /* MANDELBULB_RENDERER_HXX */
