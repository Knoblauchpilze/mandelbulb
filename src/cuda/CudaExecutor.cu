
# include "CudaExecutor.hh"

namespace utils {

  CudaExecutor::CudaExecutor(unsigned size,
                             unsigned inElementSize,
                             const utils::Sizei& bufferSize,
                             unsigned outElementSize):
    CoreObject("executor"),

    m_poolLocker(),
    m_waiter(),
    m_poolRunning(false),
    m_jobsAvailable(false),

    m_threadsLocker(),
    m_threads(),
    m_cudaAPI(utils::Vector2i(0, 0)),
    m_schedulingData(),

    m_jobsLocker(),
    m_hPrioJobs(),
    m_nPrioJobs(),
    m_lPrioJobs(),
    m_batchIndex(0u),

    m_resultsLocker(),
    m_resultsHandling(false),
    m_results(),
    m_invalidateOld(true),
    m_resWaiter(),
    m_resultsThreadLocker(),
    m_resultsHandlingThread(),

    m_paramSize(),
    m_outElemSize(),

    onJobsCompleted()
  {
    setService("cuda");

    // Check consistency.
    if (size == 0u) {
      error(
        std::string("Could not create cuda executor service"),
        std::string("Invalid thread pool size of ") + std::to_string(size)
      );
    }

    // Create the scheduling data.
    createCudaSchedulingData(size, inElementSize, bufferSize, outElementSize);

    // Create the threads associated to this object.
    createThreadPool(size);
  }

  void
  CudaExecutor::notifyJobs() {
    // Protect from concurrent accesses.
    UniqueGuard guard(m_poolLocker);
    Guard guard2(m_jobsLocker);

    // Determine whether some jobs have to be processed.
    if (!hasJobs()) {
      log(
        std::string("Tried to start jobs processing but none are defined"),
        Level::Warning
      );

      return;
    }

    // Indicate that some jobs are available.
    m_jobsAvailable = true;

    // Notify working threads.
    m_waiter.notify_all();
  }

  void
  CudaExecutor::enqueueJobs(const std::vector<CudaJobShPtr>& jobs,
                            bool invalidate)
  {
    // Protect from concurrent accesses.
    Guard guard(m_jobsLocker);

    // Invalidate jobs if needed: this include all the remaining jobs to process
    // but also notification about the ones currently being processed.
    if (invalidate) {
      m_hPrioJobs.clear();
      m_nPrioJobs.clear();
      m_lPrioJobs.clear();
    }

    {
      Guard guard(m_resultsLocker);
      m_invalidateOld = invalidate;
    }

    // Build the job by providing the batch index for these jobs.
    for (unsigned id = 0u ; id < jobs.size() ; ++id) {
      // Consistency check.
      if (jobs[id] == nullptr) {
        log(
          std::string("Discarding invalid null job ") + std::to_string(id),
          Level::Warning
        );

        continue;
      }

      std::vector<Job>* queue = nullptr;

      switch (jobs[id]->getPriority()) {
        case utils::Priority::High:
          queue = &m_hPrioJobs;
          break;
        case utils::Priority::Normal:
          queue = &m_nPrioJobs;
          break;
        case Priority::Low:
        default:
          // Assume low priority for unhandled priority.
          queue = &m_lPrioJobs;
          break;
      }

      if (queue == nullptr) {
        log(
          std::string("Could not find adequate queue for job \"") + jobs[id]->getName() + "\" with priority " +
          std::to_string(static_cast<int>(jobs[id]->getPriority())),
          Level::Error
        );

        continue;
      }

      // Check whether the job matches the internal size both for
      // parameter and output result. If this is not the case we
      // won't be able to schedule it correctly.
      if (jobs[id]->getInputDataSize() != m_paramSize) {
        log(
          std::string("Trying to submit job \"") + jobs[id]->getName() + "\" with a parameter size of " +
          std::to_string(jobs[id]->getInputDataSize()) + " while expected value is " + std::to_string(m_paramSize) +
          ", discarding it",
          utils::Level::Error
        );

        continue;
      }
      if (jobs[id]->getOutputDataSize() != m_outElemSize) {
        log(
          std::string("Trying to submit job \"") + jobs[id]->getName() + "\" with a result size of " +
          std::to_string(jobs[id]->getOutputDataSize()) + " while expected value is " + std::to_string(m_outElemSize) +
          ", discarding it",
          utils::Level::Error
        );

        continue;
      }

      queue->push_back(
        Job{
          jobs[id],
          m_batchIndex
        }
      );
    }
  }

  void
  CudaExecutor::cancelJobs() {
    // Protect from concurrent accesses.
    UniqueGuard guard(m_poolLocker);
    Guard guard2(m_jobsLocker);

    // Clear the internal queue so that no more jobs can be fetched.
    m_jobsAvailable = false;

    std::size_t count = m_hPrioJobs.size() + m_nPrioJobs.size() + m_lPrioJobs.size();
    log("Clearing " + std::to_string(count) + " remaining job(s), next batch will be " + std::to_string(m_batchIndex));

    m_hPrioJobs.clear();
    m_nPrioJobs.clear();
    m_lPrioJobs.clear();

    // Increment the batch index to mark any currently processing job
    // as invalid when it will complete.
    ++m_batchIndex;
  }

  void
  CudaExecutor::createThreadPool(unsigned size) {
    // Create the results handling thread.
    {
      Guard guard(m_resultsLocker);
      m_resultsHandling = true;
    }
    {
      Guard guard(m_resultsThreadLocker);
      m_resultsHandlingThread = std::thread(
        &CudaExecutor::resultsHandlingLoop,
        this
      );
    }

    // Start the pool.
    {
      UniqueGuard guard(m_poolLocker);
      m_poolRunning = true;
    }

    // Protect from concurrent creation of the pool.
    Guard guard(m_threadsLocker);

    // Consistency check: verify that we can provide the
    // scheduling data to each thread.
    if (m_schedulingData.size() != size) {
      error(
        std::string("Could not create cuda executor service"),
        std::string("Should create ") + std::to_string(size) + " thread(s) but only " +
        std::to_string(m_schedulingData.size()) + " cuda stream(s) are available"
      );
    }

    m_threads.resize(size);
    for (unsigned id = 0u ; id < m_threads.size() ; ++id) {
      m_threads[id] = std::thread(
        &CudaExecutor::jobFetchingLoop,
        this,
        id,
        m_schedulingData[id]
      );
    }
  }

  void
  CudaExecutor::terminateThreads() {
    m_poolLocker.lock();

    // If no threads are created, nothing to do.
    if (!m_poolRunning) {
      m_poolLocker.unlock();
      return;
    }

    // Ask the threads to stop.
    m_poolRunning = false;
    m_poolLocker.unlock();
    m_waiter.notify_all();

    // Wait for all threads to finish.
    Guard guard(m_threadsLocker);
    for (unsigned id = 0u ; id < m_threads.size() ; ++id) {
      m_threads[id].join();
    }

    m_threads.clear();

    // Now terminate the results handling thread.
    {
      m_resultsLocker.lock();

      // If the results thread is not started we don't have
      // to do anything.
      if (!m_resultsHandling) {
        m_resultsLocker.unlock();
        return;
      }

      // Stop the thread and wait for its termination.
      m_resultsHandling = false;
      m_resWaiter.notify_all();
      m_resultsLocker.unlock();

      Guard guard3(m_resultsThreadLocker);
      m_resultsHandlingThread.join();
    }
  }

  void
  CudaExecutor::createCudaSchedulingData(unsigned count,
                                         unsigned paramSize,
                                         const utils::Sizei& bufferSize,
                                         unsigned elementSize)
  {
    // Protect from concurrent accesses to the threads' data.
    Guard guard(m_threadsLocker);

    // Create resources for each needed thread.
    bool success = false;

    // Check consistency.
    if (paramSize == 0u) {
      error(
        std::string("Could not create cuda executor service"),
        std::string("Invalid parameter size of ") + std::to_string(paramSize)
      );
    }
    if (bufferSize.w() <= 0 || bufferSize.h() <= 0) {
      error(
        std::string("Could not create cuda executor service"),
        std::string("Invalid thread pool size of ") + bufferSize.toString()
      );
    }
    if (elementSize == 0u) {
      error(
        std::string("Could not create cuda executor service"),
        std::string("Invalid element size of ") + std::to_string(elementSize)
      );
    }

    // Register elements size.
    m_paramSize = paramSize;
    m_outElemSize = elementSize;

    unsigned id = 0u;
    while (id < count) {
      // Create the stream to use to schedule operations.
      cuda::stream_t stream = m_cudaAPI.create(&success);
      if (!success) {
        error(
          std::string("Could not create cuda executor service"),
          m_cudaAPI.getLastError()
        );
      }

      // Allocate the input parameters memory.
      void* paramMem = m_cudaAPI.allocate(m_paramSize, &success);
      if (!success) {
        error(
          std::string("Could not create cuda executor service"),
          m_cudaAPI.getLastError()
        );
      }

      // Allocate the output buffer.
      unsigned step = 0u;
      void* resBuffer = m_cudaAPI.allocate2D(bufferSize, elementSize, step, &success);
      if (!success || resBuffer == nullptr) {
        error(
          std::string("Could not create cuda executor service"),
          m_cudaAPI.getLastError()
        );
      }

      // Create the scheduling data and register it in the internal array.
      m_schedulingData.push_back(
        CudaSchedulingData{
          stream,
          paramMem,
          m_paramSize,
          resBuffer,
          step
        }
      );

      ++id;
    }
  }

  void
  CudaExecutor::destroyCudaSchedulingData() {
    // Protect from concurrent accesses to the threads' data.
    Guard guard(m_threadsLocker);

    // Release memory for each created stream.
    for (unsigned id = 0u ; id < m_schedulingData.size() ; ++id) {
      CudaSchedulingData& d = m_schedulingData[id];

      // Destroy the stream.
      bool success = m_cudaAPI.destroy(d.stream);
      if (!success) {
        log(
          std::string("Could not correctly destroy stream associated to thread ") + std::to_string(id) +
          " (error: \"" + m_cudaAPI.getLastError() + "\")",
          utils::Level::Error
        );
      }

      // Free the output buffer memory.
      success = m_cudaAPI.free(d.resBuffer);
      if (!success) {
        log(
          std::string("Could not correctly destroy result buffer associated to thread ") + std::to_string(id) +
          " (error: \"" + m_cudaAPI.getLastError() + "\")",
          utils::Level::Error
        );
      }

      // Free the input parameters memory.
      success = m_cudaAPI.free(d.params);
      if (!success) {
        log(
          std::string("Could not correctly destroy parameters buffer associated to thread ") + std::to_string(id) +
          " (error: \"" + m_cudaAPI.getLastError() + "\")",
          utils::Level::Error
        );
      }
    }
  }

  void
  CudaExecutor::jobFetchingLoop(unsigned threadID,
                                CudaSchedulingData gpuData)
  {
    log("Creating thread " + std::to_string(threadID) + " for thread pool", Level::Verbose);

    // Create the locker to use to wait for job to do.
    UniqueGuard tLock(m_poolLocker);

    while (m_poolRunning) {
      // Wait until either we are requested to stop or there are some
      // new jobs to process. Checking both conditions prevents us from
      // being falsely waked up (see spurious wakeups).
      m_waiter.wait(
        tLock,
        [&]() {
          return !m_poolRunning || m_jobsAvailable;
        }
      );

      // Check whether we need to process some jobs or exit the process.
      if (!m_poolRunning) {
        break;
      }

      // Attempt to retrieve a job to process.
      Job job = Job{nullptr, 0u};
      unsigned batch = 0u;
      std::size_t remaining = 0u;

      {
        Guard guard(m_jobsLocker);

        // Fetch the highest priority job available.
        if (!m_hPrioJobs.empty()) {
          job = m_hPrioJobs.back();
          m_hPrioJobs.pop_back();
        }
        else if (!m_nPrioJobs.empty()) {
          job = m_nPrioJobs.back();
          m_nPrioJobs.pop_back();
        }
        else if (!m_lPrioJobs.empty()) {
          job = m_lPrioJobs.back();
          m_lPrioJobs.pop_back();
        }

        m_jobsAvailable = hasJobs();
        batch = m_batchIndex;

        remaining = m_hPrioJobs.size() + m_nPrioJobs.size() + m_lPrioJobs.size();
      }

      // Unlock the pool mutex so that we don't block other threads while
      // processing our chunk of job. This is what effectively allows for
      // concurrency.
      tLock.unlock();

      // If we could fetch something process it.
      if (job.task != nullptr) {
        log(
          "Processing job for batch " + std::to_string(batch) + " in thread " + std::to_string(threadID) + " (remaining: " + std::to_string(remaining) + ")",
          Level::Verbose
        );

        // Execute the job and push it to the results array if it succeeded.
        if (scheduleAndExecute(*job.task, gpuData)) {
          // Notify the main thread about the result.
          UniqueGuard guard(m_resultsLocker);
          m_results.push_back(job);

          m_resWaiter.notify_one();
        }
      }

      // Once the job is done, reacquire the mutex in order to re-wait on
      // the condition variable.
      tLock.lock();
    }

    log("Terminating thread " + std::to_string(threadID) + " for scheduler pool", Level::Verbose);
  }

  void
  CudaExecutor::resultsHandlingLoop() {
    // Create the locker to use to wait for results to be processed.
    UniqueGuard rLock(m_resultsLocker);

    while (m_resultsHandling) {
      // Wait until either we are requested to stop or there are some
      // new results to analyze. Checking both conditions prevents us
      // from being falsely waked up (see spurious wakeups).
      m_resWaiter.wait(
        rLock,
        [&]() {
          return !m_resultsHandling || !m_results.empty();
        }
      );

      // Check whether we need to process some jobs or exit the process.
      if (!m_resultsHandling) {
        break;
      }

      // We want to notify listeners of the new results: to do that we
      // will copy the existing results to an internal handler, unlock
      // the mutex to allow for other results to be accumulated and
      // for longer interpretation processes to occur without ruining
      // the concurrency brought by the thread pool.
      std::vector<Job> local;
      local.swap(m_results);

      // Strip the batch index and keep only the jobs consistent with the
      // current one.
      std::vector<CudaJobShPtr> res;
      for (unsigned id = 0u ; id < local.size() ; ++id) {
        if (local[id].batch != m_batchIndex && m_invalidateOld) {
          log(std::string("Discarding job for old batch ") + std::to_string(local[id].batch) + " (current is " + std::to_string(m_batchIndex) + ")");
          continue;
        }

        res.push_back(local[id].task);
      }

      // Notify listeners.
      rLock.unlock();
      onJobsCompleted.safeEmit(
        std::string("onJobsCompleted(") + std::to_string(res.size()) + ")",
        res
      );
      rLock.lock();
    }
  }

  bool
  CudaExecutor::scheduleAndExecute(CudaJob& job,
                                   CudaSchedulingData data)
  {
    // TODO: To fully benefit from the asyncrhonicity we should allocate the
    // host memory using `cudaMallocHost`. See here (slide 11):
    // https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf

    // We need to first copy the input parameters of the job to device memory.
    bool success = m_cudaAPI.copyToDevice(data.stream, job.getInputData(), data.paramSize, data.params);
    if (!success) {
      log(
        std::string("Could not copy parameter for job ") + job.getName() +
        "err: \"" + m_cudaAPI.getLastError() + "\")",
        utils::Level::Error
      );

      return false;
    }


    // Execute the job.
    success = m_cudaAPI.launch(
      data.stream,
      []() {
        // TODO: Implementation.
        return cudaErrorInvalidDeviceFunction;
      }
    );
    if (!success) {
      log(
        std::string("Could not launch job ") + job.getName() + ("err: \"") +
        m_cudaAPI.getLastError() + "\")",
        utils::Level::Error
      );

      return false;
    }

    // Wait for the job to complete.
    success = m_cudaAPI.wait(data.stream);
    if (!success) {
      log(
        std::string("Job \"") + job.getName() + "\" failed (err: \"" +
        m_cudaAPI.getLastError() + "\")",
        utils::Level::Error
      );

      return false;
    }

    // Copy back the results to the job.
    success = m_cudaAPI.copyToHost2D(
      data.stream,
      job.getOutputSize(),
      data.resBuffer,
      data.step,
      job.getOutputData(),
      job.getOutputDataStep()
    );
    if (!success) {
      log(
        std::string("Could not copy back result for job \"") + job.getName() +
        "\" (err: \"" + m_cudaAPI.getLastError() + "\")",
        utils::Level::Error
      );

      return false;
    }
    // TODO: Implementation.

    return true;
  }

}
