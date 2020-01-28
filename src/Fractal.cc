
# include "Fractal.hh"

namespace mandelbulb {

  Fractal::Fractal(CameraShPtr cam,
                   RenderProperties rProps,
                   ShadingProperties sProps,
                   const std::vector<LightShPtr>& lights):
    utils::CoreObject(std::string("mandelbulb")),

    m_propsLocker(),

    m_camera(cam),
    m_rProps(rProps),
    m_sProps(sProps),
    m_lights(lights),

    m_computationState(State::Converged),
    m_schedule(),
    m_scheduler(
      std::make_shared<utils::CudaExecutor>(
        getWorkerThreadCount(),
        RaytracingTile::getPropsSize(),
        utils::Sizei(getTileWidth(), getTileHeight()),
        RaytracingTile::getResultSize()
      )
    ),
    m_tilesRenderedSignalID(utils::Signal<>::NoID),
    m_progress(RenderingProgress{0u, 1u}),

    m_dims(),
    m_samples(),

    onRenderingCompletionAdvanced(),
    onTilesRendered()
  {
    setService(std::string("fractal"));

    // Check consistency.
    if (m_camera == nullptr) {
      error(
        std::string("Could not create fractal"),
        std::string("Invalid null camera")
      );
    }

    build();
  }

  float
  Fractal::getPoint(const utils::Vector2i& screenCoord,
                    utils::Vector3f& worldCoord,
                    bool& hit)
  {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Assume no hit.
    hit = false;

    worldCoord.x() = std::numeric_limits<float>::lowest();
    worldCoord.y() = std::numeric_limits<float>::lowest();
    worldCoord.z() = std::numeric_limits<float>::lowest();

    float depth = -1.0f;

    // Convert to local coordinates.
    utils::Vector2f lScreen(
      screenCoord.x() + m_dims.w() / 2,
      screenCoord.y() + m_dims.h() / 2
    );

    // Consistency check
    if (lScreen.x() < 0 || lScreen.x() >= m_dims.w() ||
        lScreen.y() < 0 || lScreen.y() >= m_dims.h())
    {
      log(
        std::string("Trying to get point at coord ") + lScreen.toString() +
        " not compatible with internal camera plane size " + m_dims.toString(),
        utils::Level::Error
      );

      return -1.0f;
    }

    // Retrieve the depth at this point: this will be used both to fill
    // the return value and to get the real world coordinates of the
    // point located at said screen coordinates.
    int off = (m_dims.h() - 1u - lScreen.y()) * m_dims.w() + lScreen.x();
    depth = m_samples[off].depth;

    // Check whether we have a hit.
    if (depth < 0.0f) {
      return depth;
    }

    // We have a hit !
    hit = true;

    // Use the camera to update the real world coordinate.
    utils::Vector2f perc(
      -0.5f + 1.0f * lScreen.x() / m_dims.w(),
      -0.5f + 1.0f * lScreen.y() / m_dims.h()
    );

    utils::Vector3f dir = m_camera->getDirection(perc);
    worldCoord = m_camera->getEye() + depth * dir;

    log(
      "Screen: " + screenCoord.toString() + " dir: " + dir.toString() + ", depth: " + std::to_string(depth),
      utils::Level::Verbose
    );

    return depth;
  }

  void
  Fractal::build() {
    // Connect the thread pool `work finished` signal to the local handler.
    m_tilesRenderedSignalID = m_scheduler->onJobsCompleted.connect_member<Fractal>(
      this,
      &Fractal::handleTilesRendered
    );
  }

  void
  Fractal::scheduleRendering() {
    // Cancel existing rendering operations.
    m_scheduler->cancelJobs();

    // Generate the launch schedule.
    std::vector<RaytracingTileShPtr> tiles = generateSchedule();;

    // Convert to required pointer type.
    std::vector<utils::CudaJobShPtr> tilesAsJobs(tiles.begin(), tiles.end());

    // Return early if nothing needs to be scheduled.
    if (tilesAsJobs.empty()) {
      // Reset the internal computation state.
      m_computationState = State::Converged;

      log(
        std::string("Scheduled a rendering but no jobs where created, discarding request"),
        utils::Level::Error
      );

      return;
    }

    m_scheduler->enqueueJobs(tilesAsJobs, true);

    // Notify listeners that the progression is now empty.
    m_progress.taskProgress = 0u;
    m_progress.taskTotal = tilesAsJobs.size();

    // Start the computing.
    m_scheduler->notifyJobs();
  }

  void
  Fractal::handleTilesRendered(const std::vector<utils::CudaJobShPtr>& tiles) {
    float perc = 0.0f;

    {
      // Protect from concurrent accesses.
      Guard guard(m_propsLocker);

      // We need to copy the tiles data to the internal array. This will
      // allow to keep track of the content computed for the fractal and
      // allow the renderer to actually display it.
      for (unsigned id = 0u ; id < tiles.size() ; ++id) {
        // Convert to usable data.
        RaytracingTileShPtr tile = std::dynamic_pointer_cast<RaytracingTile>(tiles[id]);
        if (tile == nullptr) {
          log(
            std::string("Could not convert task to raytracing tile, skipping it"),
            utils::Level::Error
          );

          continue;
        }

        copyTileData(*tile);
      }

      // Add the rendered tiles to the internal progress.
      m_progress.taskProgress += tiles.size();

      log(
        "Handled " + std::to_string(tiles.size()) + " tile(s), task: " +
        std::to_string(m_progress.taskProgress) + "/" + std::to_string(m_progress.taskTotal),
        utils::Level::Verbose
      );

      // In case an iteration has been finished, schedule the next one (or stop
      // the process if we already accumulated enough iterations).
      if (m_progress.taskProgress == m_progress.taskTotal) {
        // Accumulate enough samples.
        m_computationState = State::Converged;
      }

      // Compute the global progression: we need to clamp to `100%` in case we
      // reach the last iteration and all the tasks related to it have completed
      // as in this case we didn't schedule a rendering and thus the progress is
      // still such that `m_progress.taskProgress = m_progress.taskTotal`.
      perc = std::min(1.0f, 1.0f * m_progress.taskProgress / m_progress.taskTotal);
    }

    // Notify external listeners.
    onRenderingCompletionAdvanced.safeEmit(
      std::string("onRenderingCompletionAdvanced(") + std::to_string(perc) + ")",
      perc
    );

    // Notify that some tiles have been rendered.
    onTilesRendered.safeEmit(
      std::string("onTilesRendered()")
    );
  }

  std::vector<RaytracingTileShPtr>
  Fractal::generateSchedule() {
    // Generate each tile given the internal camera and the number of tiles
    // to generate along each axis. We assume that the dimensions of the
    // camera are also represented by the internal `m_dims` array.
    std::vector<RaytracingTileShPtr> tiles;

    unsigned w = (m_dims.w() + getTileWidth() - 1u) / getTileWidth();
    unsigned h = (m_dims.h() + getTileHeight() - 1u) / getTileHeight();

    // TODO: Maybe we should persist this schedule.
    for (unsigned y = 0u ; y < h ; ++y) {
      for (unsigned x = 0u ; x < w ; ++x) {
        // The area covered by this tile can be computed from its index
        // and the dimensions of the camera plane.
        utils::Boxi area(
          -m_dims.w() / 2 + getTileWidth() / 2 + x * getTileWidth(),
          -m_dims.h() / 2 + getTileHeight() / 2 + y * getTileHeight(),
          getTileWidth(),
          getTileHeight()
        );

        // Handle cases where the dimensions of the camera plane is not a
        // perfect multiple of the tiles' dimensions.
        evenize(area);

        log(
          std::string("Generating tile ") + std::to_string(x) + "x" + std::to_string(y) + " with area " + area.toString(),
          utils::Level::Verbose
        );

        // Create the tile and register it in the schedule.
        RaytracingTileShPtr tile = std::make_shared<RaytracingTile>(area, m_dims);

        tile->setEye(m_camera->getEye());
        tile->setU(m_camera->getU());
        tile->setV(m_camera->getV());
        tile->setW(m_camera->getW());

        tile->setRenderingProps(m_rProps);
        tile->setShadingProps(m_sProps);
        tile->setLights(m_lights);

        // Register this tile.
        tiles.push_back(tile);
      }
    }

    // Return the generated schedule.
    return tiles;
  }

  void
  Fractal::evenize(utils::Boxi& area) {
    // Clamp the area so that it does not exceed the available dimensions
    // as defined in the `m_dims` attribute.
    int excessW = std::max(0, area.getRightBound() - m_dims.w() / 2);
    int excessH = std::max(0, area.getTopBound() - m_dims.h() / 2);

    if (excessW > 0) {
      if (excessW % 2 != 0) {
        log(
          std::string("Area ") + area.toString() + " will not correctly be cropped to match " + m_dims.toString(),
          utils::Level::Error
        );
      }

      area.x() -= excessW / 2;
      area.w() -= excessW;
    }
    if (excessH > 0) {
      if (excessH % 2 != 0) {
        log(
          std::string("Area ") + area.toString() + " will not correctly be cropped to match " + m_dims.toString(),
          utils::Level::Error
        );
      }

      area.y() -= excessH / 2;
      area.h() -= excessH;
    }
  }

  void
  Fractal::copyTileData(RaytracingTile& tile) {
    // Retrieve the depth map associated to the current tile and copy it to
    // the internal array. This includes copying the depth and adjusting the
    // color of each pixel.
    utils::Boxi area = tile.getArea();
    const std::vector<pixel::Data>& map = tile.getPixelsMap();

    if (map.empty()) {
      log(
        std::string("Cannot interpret tile ") + area.toString() + " with no depth map",
        utils::Level::Error
      );

      return;
    }

    // Process each pixel of the area.
    for (int y = 0 ; y < area.h() ; ++y) {
      // Compute the equivalent of the `y` coordinate both in local tile's
      // coordinate frame and in general fractal coordinate frame.
      int sOff = y * area.w();
      int lY = (y + area.getBottomBound() + m_dims.h() / 2);
      int dOff = (m_dims.h() - 1 - lY) * m_dims.w();

      for (int x = 0 ; x < area.w() ; ++x) {
        // Do the same for `x` coordinate.
        int dX = (x + area.getLeftBound() + m_dims.w() / 2);
        int off = dOff + dX;

        if (dX < 0 || dX >= m_dims.w() ||
            lY < 0 || lY >= m_dims.h())
        {
          log(
            std::string("Could not copy data at ") + std::to_string(x) + "x" + std::to_string(y) + " from " +
            area.toString() + ", local is " + std::to_string(dX) + "x" + std::to_string(lY),
            utils::Level::Error
          );

          continue;
        }

        // Save the pixel's data.
        m_samples[off].depth = map[sOff + x].depth;
        m_samples[off].color = sdl::core::engine::Color::fromRGB(
          map[sOff + x].r,
          map[sOff + x].g,
          map[sOff + x].b
        );
      }
    }
  }

}
