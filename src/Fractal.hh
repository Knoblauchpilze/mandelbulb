#ifndef    FRACTAL_HH
# define   FRACTAL_HH

# include <mutex>
# include <vector>
# include <memory>
# include <core_utils/CoreObject.hh>
# include <core_utils/Signal.hh>
# include "Camera.hh"
# include "RenderProperties.hh"
# include "RaytracingTile.hh"
# include "CudaExecutor.hh"
# include "Light.hh"

namespace mandelbulb {

  class Fractal: utils::CoreObject {
    public:

      /**
       * @brief - Create a fractal object with the sepcified viewpoint and render
       *          settings. Note that no data will be generated until the `start`
       *          method is called.
       * @param cam - the viewpoint on the fractal object.
       * @param rProps - the rendering properties for this fractal object.
       * @param sProps - the shading properties for this fractal object.
       * @param lights - the lights for this fractal object.
       */
      Fractal(CameraShPtr cam,
              RenderProperties rProps,
              ShadingProperties sProps,
              const std::vector<LightShPtr>& lights);

      virtual ~Fractal();

      /**
       * @brief - Assign new dimensions to the camera used to view this fractal
       *          object. In case the input size is not valid an error is raised.
       *          It cancels the current jobs and starts over a new rendering if
       *          the dimensions have changed compared to the current ones.
       * @param dims - the new dimensions of the camera.
       */
      void
      setCameraDims(const utils::Sizef& dims);

      /**
       * @brief - Used to update the internal properties describing lights and
       *          maybe schedule a new rendering if needed.
       * @param lights - a vector describing the lights to use to provide some
       *                 lighting on the scene.
       */
      void
      onLightsChanged(const std::vector<LightShPtr>& lights);

      /**
       * @brief - Used to rotate the camera associated to this fractal from the
       *          longitudinal and latitudinal delta given as input. This will
       *          perform the needed updates in the internal camera and then
       *          request a rendering if needed.
       *          Note that in case the axis is not valid nothing will happen.
       * @param rotations - the longitudinal and latitudinal angles to apply to
       *                    the current position of the camera.
       */
      void
      rotateCamera(const utils::Vector2f& rotations);

      /**
       * @brief - Get the current distance to the origin point of the orbit camera
       *          used to visualize the fractal.
       * @return - a value representing the distance to the center of the orbit of
       *           the camera associated to this object.
       */
      float
      getDistance() noexcept;

      /**
       * @brief - Returns the current distance estimation to the surface of the
       *          fractal.
       * @return - an estimation of the distance to the fractal given the current
       *           forward vector of the camera linked to this fractal.
       */
      float
      getDistanceEstimation() noexcept;

      /**
       * @brief - Used to change the distance of the camera to its source by the
       *          specified factor. The value will be multiplied by the current
       *          value of the distance so using a value larger than `1` will
       *          actually make the camera be farther away than it was.
       * @param dist - the new distance to assign to the viewpoint on this fractal.
       */
      void
      updateDistance(float dist);

      /**
       * @brief - Used to retrieve the data representing the current view of the
       *          camera plane. This is used by external elements wanting to get
       *          a peek at the view of the fractal given the camera settings.
       *          This method returns the internal data as a copy and return the
       *          actual dimensions of the provided vector.
       * @param colors - the list of colors for each pixel of the camera plane.
       *                 Note that the input vector is resized if needed.
       * @return - a size indicating the actual dimensions covered by the camera
       *           plane's data returned in `colors`. Allows to know how to use
       *           the data.
       */
      utils::Sizei
      getData(std::vector<sdl::core::engine::Color>& colors);

      /**
       * @brief - Used to retrieve information about the point located at the
       *          screen coordinates defined in input. In case the point does
       *          not belong to the fractal (or no information is available on
       *          it yet) the `hit` value is set to `false` and the return and
       *          `worldCoord` values should be ignored.
       * @param screenCoord - the screen coordinates for which data should be
       *                      retrieved.
       * @param worldCoord - the real world coordinate of the intersection vec
       *                     with the input coordinates.
       * @param hit - `true` if the pointed at coordinate belongs to the fractal
       *              (and thus both `worldCoord` and the return value are okay
       *              to be used) and `false` otherwise.
       * @return - the depth of the point at the specified coordinate or `-1`
       *           if the point does not belong to the fractal (so `hit` is set
       *           to `false`).
       */
      float
      getPoint(const utils::Vector2i& screenCoord,
               utils::Vector3f& worldCoord,
               bool& hit);

      /**
       * @brief - Local slot to handle a new version of the rendering properties
       *          to use. This will trigger a new rendering if needed.
       * @param props - the new set of properties to use when computing the fractal.
       */
      void
      onRenderingPropsChanged(RenderProperties props);

      /**
       * @brief - Similar to the `onRenderingPropsChanged` but handle some shading
       *          properties change. This will trigger a new rendering if needed as
       *          well.
       * @param props - the new set of shading properties to use.
       */
      void
      onShadingPropsChanged(ShadingProperties props);

    private:

      /**
       * @brief - Provide a suitable number of thread for the underlying thread
       *          pool allowing to execute the raytracing tiles.
       * @return - a number of threads to use for the internal thread pool.
       */
      static
      unsigned
      getWorkerThreadCount() noexcept;

      /**
       * @brief - Define a suitable value for the number of pixels covered along
       *          the `x` axis for a single raytracing tile.
       * @return - a value representing the number of pixels covered horizontally
       *           by a single raytracing tile.
       */
      static
      constexpr unsigned
      getTileWidth() noexcept;

      /**
       * @brief - Similar to `getTileWidth` but provides a suitable value for the
       *          number of pixels in height for each raytracing tile.
       * @return - a value representing the number of pixels covered vertically by
       *           a single raytracing tile.
       */
      static
      constexpr unsigned
      getTileHeight() noexcept;

      /**
       * @brief- Used  to connect the needed signals from the thread pool so
       *         that we can react to raytracing tiles being computed.
       */
      void
      build();

      /**
       * @brief - Used to update the internal properties and schedule a rendering
       *          by considering that the camera has changed and maybe some other
       *          internal properties as well.
       */
      void
      updateAndRender();

      /**
       * @brief - Used to generate the schedule or raytracing tiles to use to
       *          perform the rendering of the fractal given the internal cam
       *          and properties.
       *          Note that this method assumes that the internal locker is
       *          already acquired.
       */
      void
      scheduleRendering();

      /**
       * @brief - Local slot allowing to handle notifications from the internal
       *          thread pool whenever some raytracing tiles have finished their
       *          computations. Will update the internal `m_progress` data and
       *          trigger the needed signals to notify external listeners.
       * @param tiles - the tiles that have been computed.
       */
      void
      handleTilesRendered(const std::vector<utils::CudaJobShPtr>& tiles);

      /**
       * @brief - Used internally when scheduling a rendering task to create all
       *          the tiles that will be scheduled for computation. The output
       *          vector allows to covers the whole camera plane so that a full
       *          update of the fractal's data is performed.
       *          The tiles are created with the current versions of the camera
       *          and rendering props. They are saved internally as the schedule
       *          stays valid at least in terms of layout until the camera plane
       *          chanfes its dimensions again.
       *          Note that this method assumes that the internal locker has
       *          already been acquired.
       */
      void
      generateSchedule();

      /**
       * @brief - Used to make the input area not get past the dimensions of the
       *          total area defined for this fractal. This is basically used to
       *          account for cases where the total camera plane has not dims
       *          that are consistent with the desired tiles width/height. It can
       *          result in very small tiles being produced.
       * @param area - the area to evenize.
       */
      void
      evenize(utils::Boxi& area);

      /**
       * @brief - Used to copy back the data from the input vector into the internal
       *          array. The data is assumed to represent the area provided as input
       *          and should contain both the depth and the color to assign to the
       *          points.
       * @param tile - the raytracing tile representing the data to copy.
       */
      void
      copyTileData(RaytracingTile& tile);

    private:

      /**
       * @brief - Describe the possible state for the computation. Basically the
       *          computation of the fractal is an iterative process which uses
       *          several individual iterations allowing to accumulate data for
       *          each pixel.
       *          Whenever the render properties or the viewpoint are changed it
       *          should trigger a new rendering step.
       *          On the other hand when the required number of iterations has
       *          been computed there's no need to continue accumulating results
       *          and thus we shouldn't continue scheduling jobs.
       */
      enum class State {
        Converged,
        Accumulating
      };

      /**
       * @brief - Convenience structure regrouping the progress of the current
       *          iteration and the overall progress of the rendering job.
       */
      struct RenderingProgress {
        unsigned taskProgress; ///< Holds the number of finished tiles for
                               ///< the current iteration.
        unsigned taskTotal;    ///< Holds the number of tiles generated for
                               ///< this single iteration.
      };

      /**
       * @brief - Convenience structure representing the result of a pixel of
       *           the fractal. This contains information about the number of
       *           iterations accumulated for this sample along with some data
       *           regarding the depth and normal of the element.
       */
      struct Sample {
        float depth;                    ///< The average depth of this pixel (based on the number
                                        ///< iterations accumulated).
        sdl::core::engine::Color color; ///< The color for this sample. Just like the depth it is
                                        ///< averaged with the iteration count.
      };

      /**
       * @brief - Protects this object from concurrent accesses. This is
       *          used to provide some way to guarantee that we're not
       *          trying to display the content of the fractal while it
       *          is being updated after a successful raytracing operation.
       */
      std::mutex m_propsLocker;

      /**
       * @brief - The current camera used to look at he fractal. Passed on
       *          to raytracing tiles so that they know what needs to be
       *          computed.
       *          This is updated with the actions of the user through the
       *          `rotate` and `translate` methods so that we are allowed
       *          to move around the fractal.
       */
      CameraShPtr m_camera;

      /**
       * @brief - The properties to use when rendering the fractal. This
       *          is passed on to the raytracing tiles that can be created
       *          by this object so that the rendering phase knows the
       *          level of accuracy to apply.
       */
      RenderProperties m_rProps;

      /**
       * @brief - The shading properties to use when rendering the fractal. It
       *          defines common values such as the color of the fractal with
       *          no lights on, the no data color and some post-processing vals
       *          which are applied to make the final image prettier.
       */
      ShadingProperties m_sProps;

      /**
       * @brief - The list of lights used to illuminate the scene. Received
       *          from the dedicated control panel and passed on to the tiles
       *          used to perform the computations.
       */
      std::vector<LightShPtr> m_lights;

      /**
       * @brief - Describes the current computation state for this fractal.
       *          It allows to determine what to do when the last tile is
       *          declared finished: we can either schedule some more tiles
       *          if the iterations count is not yet sufficient in regard
       *          of the properties or stop the process.
       */
      State m_computationState;

      /**
       * @brief - Describe the array of tiles to be executed to produce a
       *          rendering of the fractal. As the tiles are mostly linked
       *          to a camera dimensions and not to the position of the
       *          camera we figured that it might be interesting to persist
       *          this rather than creating it each time a new rendering is
       *          requested.
       *          The class provides convenience methods to populate the
       *          varying parts (namely the rendering properties and the
       *          lights) so it got us covered.
       *          The only way this vector needs to be rebuilt is when the
       *          dimensions of the camera used to visualize the fractal
       *          are changed.
       */
      std::vector<RaytracingTileShPtr> m_schedule;

      /**
       * @brief - The scheduler allowing to launch the jobs needed to update
       *          all the individual parts of the camera plane using a thread
       *          pool. It handles the scheduling of individual tiles (which
       *          are computed and generated by this object) and notify of the
       *          completion of each one.
       */
      utils::CudaExecutorShPtr m_scheduler;

      /**
       * @brief - Index of the signal provided by the scheduler to notify of
       *          rendered tiles. Will be disconnected before destroying the
       *          object.
       */
      int m_tilesRenderedSignalID;

      /**
       * @brief - Holds the progress of the current rendering process. Note
       *          that the data of this attribute is irrelevant if the state
       *          of the computation indicates that the accumulation ended.
       */
      RenderingProgress m_progress;

      /**
       * @brief - Holds the dimensions of the internal `m_samples` array. This
       *          should more or less reflect the camera plane used to visualize
       *          this fractal object and allows to determine how to interpret
       *          the `m_samples` array.
       */
      utils::Sizei m_dims;

      /**
       * @brief - The actual data computed for this fractal. In order to perform
       *          the raytracing we cast some rays along the camera plane and
       *          accumulate the depth at which the fractal was encountered (if
       *          it hits the fractal) along with some other information. We can
       *          then query this array to build a visual representation of the
       *          fractal through the `getPixels` method.
       */
      std::vector<Sample> m_samples;

    public:

      /**
       * @brief - Signal notifying external listeners that the rendering of the
       *          fractal with the current viewpoint and properties has progressed
       *          to the specified value.
       */
      utils::Signal<float> onRenderingCompletionAdvanced;

      /**
       * @brief - Signal notifying external listeners that some tiles have been
       *          rendered.
       */
      utils::Signal<> onTilesRendered;
  };

  using FractalShPtr = std::shared_ptr<Fractal>;
}

# include "Fractal.hxx"

#endif    /* FRACTAL_HH */
