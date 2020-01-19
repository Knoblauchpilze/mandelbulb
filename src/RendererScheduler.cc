
# include "RendererScheduler.hh"
// TODO: Include the actual implementation of the job to process.

namespace mandelbulb {

  RendererScheduler::RendererScheduler():
    utils::CoreObject(std::string("mandelbulb_scheduler")),

    m_propsLocker(),

    m_scheduler(std::make_shared<utils::ThreadPool>(getWorkerThreadCount())),
    m_computation(Rendering::Converged),
    m_taskProgress(0u),
    m_taskTotal(1u),

    onTileRendered()
  {
    setService("scheduler");

    build();
  }

  void
  RendererScheduler::start() {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Check whether we're already accumulating results.
    if (m_computation == Rendering::Accumulating) {
      return;
    }

    // Assign the new state.
    m_computation = Rendering::Accumulating;

    scheduleRendering();
  }

  void
  RendererScheduler::stop() {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Check whether the rendering has already converged
    // (and thus stopped).
    if (m_computation == Rendering::Converged) {
      return;
    }

    // Stop the accumulation of results once all the current jobs
    // have finished.
    m_computation = Rendering::Converged;
  }

  void
  RendererScheduler::build() {
    // Connect the results provider signal of the thread pool to the local slot.
    m_scheduler->onJobsCompleted.connect_member<RendererScheduler>(
      this,
      &RendererScheduler::handleTilesRenderer
    );

    // Disable logging for the scheduler.
    m_scheduler->allowLog(false);
  }

  void
  RendererScheduler::scheduleRendering() {
    // Cancel existing rendering operations.
    m_scheduler->cancelJobs();

    // Generate the launch schedule.
    // TODO: Generate schedule.

    // Convert to required pointer type.
    std::vector<utils::AsynchronousJobShPtr> tilesAsJobs;//(tiles.begin(), tiles.end());

    // Return early if nothing needs to be scheduled.
    if (tilesAsJobs.empty()) {
      // Reset the internal computation state.
      m_computation = Rendering::Converged;

      log(
        std::string("Scheduled a rendering but no jobs where created, discarding request"),
        utils::Level::Error
      );

      // TODO: See if there's anything more to do.
      return;
    }

    m_scheduler->enqueueJobs(tilesAsJobs, false);

    // Notify listeners that the progression is now `0`.
    m_taskProgress = 0u;
    m_taskTotal = tilesAsJobs.size();

    // Start the computing.
    m_scheduler->notifyJobs();
  }

  void
  RendererScheduler::handleTilesRenderer(const std::vector<utils::AsynchronousJobShPtr>& tiles) {
    // Protect from concurrent accesses.
    Guard guard(m_propsLocker);

    // Append the number of tiles to the internal count.
    m_taskProgress += tiles.size();

    // In case all the tiles have been computed for this iteration, notify external
    // listeners. Otherwise wait for the iteration to complete.
    if (m_taskProgress == m_taskTotal) {
      // TODO: Handle notification of the tiles computed (top schedule the repaint).

      // Check whether we should schedule a new iteration based on the status of
      // the computation. We will also update the computation state accordingly.
      switch (m_computation) {
        case Rendering::Accumulating:
          scheduleRendering();
          break;
        case Rendering::Converged:
          // Nothing to do, the rendering has converged (or reached the maximum
          // allowed iterations count) or is already stopped we will not start
          // a new rendering nor modify the state.
        default:
          // Same behavior as if the computation was stopped.
          break;
      }
    }
  }

}
