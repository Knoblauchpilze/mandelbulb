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

    // Update the rendering options if needed.
    if (m_fractal != nullptr) {
      m_fractal->setCameraDims(window.toSize());
    }
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

    // We need to both retrieve the depth at the position of the mouse and
    // the real world coordinate of the point. Note that in order to send
    // the info that the coordinates are invalid we will use the minimum
    // value of a `float`.
    bool hit = false;
    utils::Vector3f wCoords(
      std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::lowest()
    );

    // Convert the position to internal coordinates.
    utils::Vector2f fConv = mapFromGlobal(e.getMousePosition());
    utils::Vector2i conv(
      static_cast<int>(std::round(fConv.x())),
      static_cast<int>(std::round(fConv.y()))
    );

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

      // Protect from concurrent accesses.
      Guard guard(m_propsLocker);

      // Perform the zoom in/out operation.
      float factor = motion.y() > 0 ? getDefaultZoomInFactor() : getDefaultZoomOutFactor();
      
      m_fractal->updateDistance(factor);
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

    gr->setColorAt(0.0f, sdl::core::engine::Color::NamedColor::Red);
    gr->setColorAt(1.0f, sdl::core::engine::Color::NamedColor::Blue);

    return gr;
  }

  float
  MandelbulbRenderer::getDefaultPaletteRange() noexcept {
    return 10.0f;
  }

  inline
  sdl::core::engine::Color
  MandelbulbRenderer::getNoDataColor() noexcept {
    return sdl::core::engine::Color::NamedColor::Cyan;
  }

  inline
  void
  MandelbulbRenderer::onTilesRendered() {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Create and post a repaint event.
    postEvent(
      std::make_shared<sdl::core::engine::PaintEvent>(
        mapToGlobal(getRenderingArea(), false)
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

}

#endif    /* MANDELBULB_RENDERER_HXX */
