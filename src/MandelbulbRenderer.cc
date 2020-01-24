
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

    // TODO: Provide customization of palette.
    m_palette(RenderPalette{
      generateDefaultPalette(),
      getDefaultPaletteRange()
    }),

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
    Guard guard(m_propsLocker);

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
    float angle = 0.0f;
    utils::Vector3f axis(0.0f, 0.0f, 1.0f);

    float delta = getArrowKeyRotationAngle();

    if (e.getRawKey() == sdl::core::engine::RawKey::Left) {
      move = true;
      angle -= delta;
    }
    if (e.getRawKey() == sdl::core::engine::RawKey::Right) {
      move = true;
      angle += delta;
    }

    // Schedule a scrolling if some motion has been detected.
    if (move) {
      Guard guard(m_propsLocker);

      m_fractal->rotateCamera(utils::Vector2f(angle, 0.0f));

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
    std::vector<float> depths;
    utils::Sizei s = m_fractal->getData(depths);

    // Convert it into colors.
    std::vector<sdl::core::engine::Color> colors(s.area(), getNoDataColor());

    for (int y = 0 ; y < s.h() ; ++y) {
      int off = y * s.w();

      for (int x = 0 ; x < s.w() ; ++x) {
        // Retrieve a color if this pixel contains data.
        if (depths[off + x] >= 0.0f) {
          // Convert the depth in the range.
          float depth = std::fmod(depths[off + x], m_palette.range) / m_palette.range;

          // Assign the color using the palette.
          // Perform an inversion of the `y` axis so that the generated
          // image is not upside down.
          int rOff = (s.h() - 1 - y) * s.w();
          colors[rOff + x] = m_palette.palette->getColorAt(depth);
        }
      }
    }

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
