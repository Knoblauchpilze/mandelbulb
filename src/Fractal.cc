
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
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Add the rendered tiles to the internal progress.
    m_progress.taskProgress += tiles.size();

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

    // Compute the global progression.
    float perc = 1.0f *
      (m_progress.iterationProgress * m_progress.taskTotal + m_progress.taskProgress) /
      (m_progress.desiredIterations * m_progress.taskTotal)
    ;

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
    // TODO: Implementation.
    return std::vector<RaytracingTileShPtr>();
  }

}
