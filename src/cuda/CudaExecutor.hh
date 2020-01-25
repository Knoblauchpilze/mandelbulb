#ifndef    CUDA_EXECUTOR_HH
# define   CUDA_EXECUTOR_HH

# include <core_utils/CoreObject.hh>
# include <maths_utils/Size.hh>
# include <mutex>
# include <vector>
# include <thread>
# include <condition_variable>
# include <core_utils/Signal.hh>
# include "CudaJob.hh"
# include "CudaWrapper.cuh"

namespace utils {

  class CudaExecutor: public CoreObject {
    public:

    /**
     * @brief - Create a new cuda executor with the specified thread pool size.
     *          This size will be used to create dedicated cuda streams that are
     *          then used to schedule several kernels concurrently and thus use
     *          the scheduling provided by the cuda API.
     *          In addition to the number of threads the executor is also given
     *          parameters to describe the expected results of the computations.
     *          Indeed each job will be given a certain amount of memory to work
     *          with (both to retrieve input parameters and to save results) and
     *          we need a description of the amount of the memory needed.
     *          Buffers will be allocated once per kernel and will be persisted
     *          through the calls so as not to waste processing time allocating
     *          a buffer for a single use.
     * @param size - the number of cuda streams to create for this pool.
     * @param bufferSize - the size of the buffers passed to cuda kernels using
     *                     a single element metric: the total size is computed
     *                     from this value and the `elementSize` value.
     * @param elementSize - the size in bytes of a single element of the output
     *                      buffer. The total size to allocate is computed from
     *                      both this value and `bufferSize` and is persisted
     *                      through launches.
     */
    CudaExecutor(unsigned size,
                  const Sizei& bufferSize,
                  unsigned elementSize);

    /**
     * @brief - Used to destroy the pool and terminate all the threads used to
     *          process the jobs. Note that we wait for job's completion before
     *          destroying the threads so this can take some time.
     */
    virtual ~CudaExecutor();

    /**
     * @brief - Used to notify that some jobs have been inserted into the
     *          internal queue and that threads can start the computation.
     */
    void
    notifyJobs();

    /**
     * @brief - Used to enqueue the list of jobs provided in input into the internal
     *          queue. The user should call the `notifyJobs` method to actually start
     *          the processing.
     *          One can choose whether these jobs invalidate the work that is being
     *          processes right now or on the other hand if it does not impact it.
     *          The second boolean, if set to `true` will prevent the notification of
     *          jobs from previous batches through the `onJobsCompleted` signal. If
     *          the value is set to `false`, the user will still be able to get some
     *          notifications about previously submitted jobs.
     * @param jobs - the list of jobs to enqueue.
     * @param invalidate - prevent notification of jobs from previous batches if set
     *                     to `true`.
     */
    void
    enqueueJobs(const std::vector<CudaJobShPtr>& jobs,
                bool invalidate);

    /**
     * @brief - Used to cancel any existing jobs being processed for this scheduler.
     *          This function is needed in order to be able to call `enqueueJobs` again.
     */
    void
    cancelJobs();

  private:

    /**
     * @brief - Used to create the thread pool used by this scheduler to perform the
     *          user's computations.
     *          The goal of this method is to create right away the cuda memory areas
     *          that will be used by each thread of processing so that it is readily
     *          available when jobs are submitted. Note that the threads are started
     *          but wait for `notifyJobs` to actually start the processing.
     * @param size - the number of thread(s) (and cuda stream(s)) to create for this
     *               pool.
     */
    void
    createThreadPool(unsigned size);

    /**
     * @brief - Used to terminate the threads associated to the thread pool. This is
     *          typically called upon destroying the scheduler. Note that the memory
     *          allocated on the gpu device will also be freed.
     */
    void
    terminateThreads();

    /**
     * @brief - Used to create the scheduling data used by the cuda kernels to perform
     *          the computations. This include both the streams which will be used to
     *          execute concurrently the kernels but also the needed resources such as
     *          the buffers to store the computations' results and the input data.
     *          The results are saved into the `m_schedulingData` array so that they
     *          can both be passed on to the executing threads and freed upon stopping
     *          the executor service.
     * @param count - the number of resources to create. This usually corresponds to
     *                the number of threads to create to process the jobs.
     * @param buffer - the number of elements in the output buffer used by the jobs to
     *                 communicate back the results.
     * @param elementSize - the size in bytes of a single element of the output buffer.
     */
    void
    createCudaSchedulingData(unsigned count,
                             const utils::Sizei& bufferSize,
                             unsigned elementSize);

    /**
     * @brief - Used to call the cuda API to free the resources allocated to perform
     *          the execution of jobs. This include the output buffers along with the
     *          input memory to pass arguments to the cuda threads.
     */
    void
    destroyCudaSchedulingData();

    /**
     * @brief - Used as a thread loop method when creating the pool. This method will
     *          be executed by each individual thread and handles the querying of the
     *          jobs, processing itself, notification of the results and termination
     *          when needed.
     *          This method takes an identifier that is used to identify the jobs that
     *          are processed by the thread (and the cuda stream).
     * @param threadID - a provided counter identifying this thread. Nothing fancy but
     *                   it allows to easily determine from which thread the completed
     *                   jobs come from.
     */
    void
    jobFetchingLoop(unsigned threadID);

    /**
     * @brief - Used as the main loop method when creating the threads to handle the
     *          results produced by the threads and notify it somehow to external
     *          listeners.
     *          The results are analyzed to determine whether they belong to the batch
     *          currently being processed: in any other case they are discarded.
     */
    void
    resultsHandlingLoop();

    /**
     * @brief - Used to determine whether any jobs at all are registered. Scans all
     *          the priority queues and return `true` if at least one of them is not
     *          empty. Assumes that the locker used to protect the jobs' queues is
     *          already acquired.
     * @return - `true` if at least one job is subtmitted (no matter the priority).
     */
    bool
    hasJobs() const noexcept;

  private:

    /**
     * @brief- Convenience define to refer to the type of locker to protect the pool
     *         running status from concurrent accesses.
     */
    using Mutex = std::mutex;

    /**
     * @brief - Convenience define to refer to a unique lock on the mutex used to
     *          protect the pool's running status.
     */
    using UniqueGuard = std::unique_lock<Mutex>;

    /**
     * @brief - Convenience structure representing a job and the corresponding batch
     *          index. This allows to identify whether a result is linked to the current
     *          batch or to an old one.
     */
    struct Job {
      CudaJobShPtr task;
      unsigned batch;
    };

    /**
     * @brief - This structure is used to handle the creation of the needed resources
     *          to compute the jobs. This include the input data and the output buffer
     *          which will be persisted between cuda launches.
     *          Such a structrue is passed on to each created thread which will then
     *          use it to schedule the jobs that are fed to it.
     *          The creation and deletion of these resoruces is handled by the executor
     *          iteself and not the threads.
     */
    struct CudaSchedulingData {
      cuda::stream_t stream;
      void* resBuffer;
      void* paramsBuffer;
    };

    /**
     * @brief - A mutex protecting concurrent accesses to the threads composing the
     *          pool. Typically used to start or stop the thread pool.
     */
    Mutex m_poolLocker;

    /**
     * @brief - Condition variable used to put threads of the pool to sleep as long
     *          as no jobs are provided and the pool does not need to be terminated.
     */
    std::condition_variable m_waiter;

    /**
     * @brief - Keep track of whether the pool is running. As long as this value is
     *          `true` individual threads can continue fetching information and wait
     *          for jobs.
     */
    bool m_poolRunning;

    /**
     * @brief - Indicates whether there are some jobs to process. A `true` value
     *          tells that the internal queue for computing jobs has at least one
     *          value. We protect this boolean behind the same locker (i.e. the
     *          `m_poolLocker` one) as the `m_poolRunning` boolean because we want
     *          threads to be notified either when the pool needs to be terminated
     *          or when some new jobs are available.
     */
    bool m_jobsAvailable;

    /**
     * @brief - Protect concurrent accesses to the array of threads.
     */
    Mutex m_threadsLocker;

    /**
     * @brief - The threads used by the pool. When the pool is up and running there
     *          should be `getThreadPoolSize` threads registered in the vector. A
     *          termination of the pool destroys the thread but in general they should
     *          not be accessed directly.
     */
    std::vector<std::thread> m_threads;

    /**
     * @brief - Convenience object allowing to have a nice interface with the cuda API
     *          by wrapping most function call and providing easy error checking. Note
     *          that most of the creation of cuda resources should be handled by this
     *          object in order to have a single point of access to the underlying API.
     */
    CudaWrapper m_cudaAPI;

    /**
     * @brief - An array holding the cuda resources created to perform the scheduling
     *          of jobs. Note that usually this array contains as many elements as there
     *          are threads in the `m_threads` vector.
     *          This data is passed on to each thread so that they can actually schedule
     *          and execute the cuda jobs on the gpu.
     *          The lifecycle of this data is guaranteed to be at least as long as the
     *          threads using it so it shouldn't be a concern to access it while threads
     *          are running.
     */
    std::vector<CudaSchedulingData> m_schedulingData;

    /**
     * @brief - Protect concurrent accesses to the jobs queue and related properties.
     */
    Mutex m_jobsLocker;

    /**
     * @brief - The list of jobs currently available for processing. Contains all the
     *          high priority jobs.
     */
    std::vector<Job> m_hPrioJobs;

    /**
     * @brief - Similar to the `m_hPrioJobs` queue but contains normal priority jobs.
     */
    std::vector<Job> m_nPrioJobs;

    /**
     * @brief - Similar to the `m_hPrioJobs` queue but contains the low priority jobs.
     */
    std::vector<Job> m_lPrioJobs;

    /**
     * @brief - An index identifying the current batch of jobs being fed to the
     *          threads. Any completion related to another batch will be discarded
     *          as it's probably irrelevant anymore.
     */
    unsigned m_batchIndex;

    /**
     * @brief - Protects the access to the results properties (thread and waiting
     *          condition).
     */
    Mutex m_resultsLocker;

    /**
     * @brief - Indicates whether the results handling process should still be occuring.
     *          This allows to terminate gracefully the results thread.
     */
    bool m_resultsHandling;

    /**
     * @brief - The list of jobs already computed, available for analysis.
     */
    std::vector<Job> m_results;

    /**
     * @brief - Defines whether the job related to an old batch should be considered valid
     *          or if no notification should be produced for them.
     */
    bool m_invalidateOld;

    /**
     * @brief - Waiting condition to communicate results to the dedicated thread.
     */
    std::condition_variable m_resWaiter;

    /**
     * @brief - A mutex protecting the results handling thread.
     */
    Mutex m_resultsThreadLocker;

    /**
     * @brief - A thread used to handle the results communicated by other threads of the
     *          pool. This one should be terminated after all the other threads otherwise
     *          we could hang the program.
     */
    std::thread m_resultsHandlingThread;

  public:

    /**
     * @brief - This signal is emitted by the scheduler as soon as some jobs have been
     *          successfully rendered by the thread pool.
     *          Any listener whishing to update itself with the results of the process
     *          can register on this signal and be notified when that happens.
     */
    Signal<const std::vector<CudaJobShPtr>&> onJobsCompleted;
  };

}

# include "CudaExecutor.hxx"

#endif    /* CUDA_EXECUTOR_HH */
