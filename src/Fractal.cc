
# include "Fractal.hh"

namespace mandelbulb {

  Fractal::Fractal(CameraShPtr cam,
                   RenderProperties props):
    utils::CoreObject(std::string("mandelbulb")),

    m_propsLocker(),

    m_camera(cam),
    m_props(props),

    m_computationState(State::Converged),
    m_scheduler(
      std::make_shared<utils::ThreadPool>(
        getWorkerThreadCount()
      )
    ),
    m_progress(RenderingProgress{
      // Current iteration
      0u,
      1u,

      // Iterations count
      0u,
      1u
    }),

    m_dims(),
    m_samples(),

    onCameraChanged(),
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

  void
  Fractal::build() {
    // Connect the thread pool `work finished` signal to the local handler.
    m_scheduler->onJobsCompleted.connect_member<Fractal>(
      this,
      &Fractal::handleTilesRendered
    );

    // Disable logging for the scheduler.
    m_scheduler->allowLog(false);
  }

  void
  Fractal::scheduleRendering(bool reset) {
    // Cancel existing rendering operations.
    m_scheduler->cancelJobs();

    // Generate the launch schedule.
    std::vector<RaytracingTileShPtr> tiles = generateSchedule();;

    // Convert to required pointer type.
    std::vector<utils::AsynchronousJobShPtr> tilesAsJobs(tiles.begin(), tiles.end());

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

    m_scheduler->enqueueJobs(tilesAsJobs, false);

    // Notify listeners that the progression is now empty.
    m_progress.taskProgress = 0u;
    m_progress.taskTotal = tilesAsJobs.size();

    // Reset iterations if needed.
    if (reset) {
      m_progress.iterationProgress = 0u;
      m_progress.desiredIterations = m_props.iterations;
    }

    // Start the computing.
    m_scheduler->notifyJobs();
  }

  void
  Fractal::handleTilesRendered(const std::vector<utils::AsynchronousJobShPtr>& tiles) {
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

        // Retrieve the depth map associated to the current tile and copy it to
        // the internal array. This includes copying the depth and adjusting the
        // iterations count.
        utils::Boxi area = tile->getArea();
        const std::vector<float>& map = tile->getDepthMap();

        for (int y = 0 ; y < area.h() ; ++y) {
          // Compute the equivalent of the `y` coordinate both in local tile's
          // coordinate frame and in general fractal coordinate frame.
          int sOff = y * area.w();
          int lY = (y + area.getBottomBound() + m_dims.h() / 2);
          int dOff = lY * m_dims.w();

          for (int x = 0 ; x < area.w() ; ++x) {
            // Do the same for `x` coordinate.
            int dX = (x + area.getLeftBound() + m_dims.w() / 2);

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

            float depth = map[sOff + x];
            if (depth >= 0.0f) {
              if (m_samples[dOff + dX].iter == 0u) {
                m_samples[dOff + dX].depth = depth;
              }
              else {
                m_samples[dOff + dX].depth += depth;
              }
            }

            ++m_samples[dOff + dX].iter;
          }
        }
      }

      // Add the rendered tiles to the internal progress.
      m_progress.taskProgress += tiles.size();

      log(
        "Handled " + std::to_string(tiles.size()) + " tile(s), task: " +
        std::to_string(m_progress.taskProgress) + "/" + std::to_string(m_progress.taskTotal) + ", " +
        " iteration: " + std::to_string(m_progress.iterationProgress) + "/" + std::to_string(m_progress.desiredIterations),
        utils::Level::Verbose
      );

      // In case an iteration has been finished, schedule the next one (or stop
      // the process if we already accumulated enough iterations).
      if (m_progress.taskProgress == m_progress.taskTotal) {
        ++m_progress.iterationProgress;

        // Check whether we should start a new iteration.
        if (m_progress.iterationProgress == m_progress.desiredIterations) {
          // No need to start a new iteration, we already accumulate enough. We
          // need to reset the local state.
          m_computationState = State::Converged;
        }
        else {
          // Schedule a new iteration: we don't want to erase the complete
          // progression as it's still the same rendering.
          scheduleRendering(false);
        }
      }
    }

    // Compute the global progression: we need to clamp to `100%` in case we
    // reach the last iteration and all the tasks related to it have completed
    // as in this case we didn't schedule a rendering and thus the progress is
    // still such that `m_progress.taskProgress = m_progress.taskTotal`.
    float perc = std::min(1.0f, 1.0f *
      (m_progress.iterationProgress * m_progress.taskTotal + m_progress.taskProgress) /
      (m_progress.desiredIterations * m_progress.taskTotal)
    );

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

        // Clamp the area so that it does not exceed the available dimensions
        // as defined in the `m_dims` attribute.
        int excessW = std::max(0, area.getRightBound() - m_dims.w() / 2);
        int excessH = std::max(0, area.getTopBound() - m_dims.h() / 2);

        if (excessW > 0) {
          if (excessW % 2 != 0) {
            log(
              std::string("Area ") + area.toString() + " will not correctly be cropped to match " +
              m_dims.toString(),
              utils::Level::Error
            );
          }

          area.x() -= excessW / 2;
          area.w() -= excessW;
        }
        if (excessH > 0) {
          if (excessH % 2 != 0) {
            log(
              std::string("Area ") + area.toString() + " will not correctly be cropped to match " +
              m_dims.toString(),
              utils::Level::Error
            );
          }

          area.y() -= excessH / 2;
          area.h() -= excessH;
        }

        log(
          std::string("Generating tile ") + std::to_string(x) + "x" + std::to_string(y) +
          " with area " + area.toString(),
          utils::Level::Verbose
        );

        // Create the tile and register it in the schedule.
        tiles.push_back(
          std::make_shared<RaytracingTile>(
            m_camera->getEye(),
            m_camera->getU(),
            m_camera->getV(),
            m_camera->getW(),
            m_dims,
            area
          )
        );
      }
    }

    // Return the generated schedule.
    return tiles;
  }

}
