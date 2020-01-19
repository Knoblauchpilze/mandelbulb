#ifndef    RENDERER_SCHEDULER_HH
# define   RENDERER_SCHEDULER_HH

# include <mutex>
# include <memory>
# include <core_utils/CoreObject.hh>
# include <maths_utils/Box.hh>
# include <core_utils/ThreadPool.hh>

namespace mandelbulb {

  class RendererScheduler: utils::CoreObject {
    public:

      RendererScheduler();

      ~RendererScheduler();

      void
      start();

      void
      stop();

    private:

      static
      unsigned
      getWorkerThreadCount() noexcept;

      void
      build();

      void
      scheduleRendering();

      void
      handleTilesRenderer(const std::vector<utils::AsynchronousJobShPtr>& tiles);

    private:

      /**
       * @brief - Describe the possible state for the computation.
       */
      enum class Rendering {
        Converged,
        Accumulating
      };

      std::mutex m_propsLocker;

      utils::ThreadPoolShPtr m_scheduler;

      Rendering m_computation;

      unsigned m_taskProgress;

      unsigned m_taskTotal;

    public:

      utils::Signal<utils::Boxf> onTileRendered;
  };

  using RendererSchedulerShPtr = std::shared_ptr<RendererScheduler>;
}

# include "RendererScheduler.hxx"

#endif    /* RENDERER_SCHEDULER_HH */
