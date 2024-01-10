
# include "MandelbulbRenderer.hh"

namespace mandelbulb {

  MandelbulbRenderer::MandelbulbRenderer(FractalShPtr fractal,
                                         const utils::Sizef& hint,
                                         sdl::core::SdlWidget* parent):
    sdl::graphic::ScrollableWidget(std::string("renderer"),
                                   parent,
                                   hint),

    m_propsLocker(),
    m_fractal(fractal),
    m_tilesRenderedSignalID(utils::Signal<>::NoID),

    m_tex(),
    m_tilesRendered(false),

    onCoordinatesChanged(),
    onDepthChanged()
  {
    setService(std::string("mandelbulb"));

    // Consistency check.
    if (m_fractal == nullptr) {
      error(
        std::string("Could not create mandelbulb renderer"),
        std::string("Invalid null fractal object to display")
      );
    }

    build();
  }

  void
  MandelbulbRenderer::drawContentPrivate(const utils::Uuid& uuid,
                                         const utils::Boxf& area)
  {
    // Protect from concurrent accesses.
    const std::lock_guard guard(m_propsLocker);

    // Re-render the fractal if needed.
    if (tilesChanged()) {
      // Load the fractal's data.
      loadTiles();

      // No need to udpate the rendering on the next display.
      m_tilesRendered = false;
    }

    // Check whether there's something to display.
    if (!m_tex.valid()) {
      return;
    }

    // Convert the area to local so that we blit only the right area of
    // the texture representing the fractal.
    utils::Boxf thisArea = LayoutItem::getRenderingArea().toOrigin();
    utils::Sizef sizeEnv = getEngine().queryTexture(uuid);
    utils::Sizef texSize = getEngine().queryTexture(m_tex);

    utils::Boxf srcArea = thisArea.intersect(area);
    utils::Boxf dstArea = thisArea.intersect(area);

    utils::Boxf srcEngine = convertToEngineFormat(srcArea, texSize);
    utils::Boxf dstEngine = convertToEngineFormat(dstArea, sizeEnv);

    getEngine().drawTexture(m_tex, &srcEngine, &uuid, &dstEngine);
  }

  bool
  MandelbulbRenderer::keyPressEvent(const sdl::core::engine::KeyEvent& e) {
    // Check for arrow keys.
    bool move = false;
    float theta = 0.0f;
    float phi = 0.0f;

    float delta = getArrowKeyRotationAngle();

    if (e.getRawKey() == sdl::core::engine::RawKey::Left) {
      move = true;
      theta += delta;
    }
    if (e.getRawKey() == sdl::core::engine::RawKey::Right) {
      move = true;
      theta -= delta;
    }
    if (e.getRawKey() == sdl::core::engine::RawKey::Up) {
      move = true;
      phi += delta;
    }
    if (e.getRawKey() == sdl::core::engine::RawKey::Down) {
      move = true;
      phi -= delta;
    }

    // Schedule a scrolling if some motion has been detected.
    if (move) {
      const std::lock_guard guard(m_propsLocker);

      m_fractal->rotateCamera(utils::Vector2f(theta, phi));

      requestRepaint();
    }

    // Use the base handler to provide a return value.
    return sdl::graphic::ScrollableWidget::keyPressEvent(e);
  }

  void
  MandelbulbRenderer::build() {
    // Connect the fractal signal indicating that some tiles have been rendered to the
    // local slot allowing to schedule a repaint.
    m_tilesRenderedSignalID = m_fractal->onTilesRendered.connect_member<MandelbulbRenderer>(
      this,
      &MandelbulbRenderer::onTilesRendered
    );
  }

  void
  MandelbulbRenderer::loadTiles() {
    // Clear any existing texture representing the tiles.
    clearTiles();

    // Consistency check.
    if (m_fractal == nullptr) {
      return;
    }

    // Retrieve the data from the fractal object.
    std::vector<sdl::core::engine::Color> colors;
    utils::Sizei s = m_fractal->getData(colors);

    // Create the engine from this data.
    sdl::core::engine::BrushShPtr brush = std::make_shared<sdl::core::engine::Brush>(
      std::string("fractal_brush"),
      false
    );

    brush->createFromRaw(s, colors);

    // Use the brush to create a texture.
    m_tex = getEngine().createTextureFromBrush(brush);

    if (!m_tex.valid()) {
      error(
        std::string("Could not create texture to represent fractal"),
        std::string("Failed to transform brush into texture")
      );
    }
  }

}
