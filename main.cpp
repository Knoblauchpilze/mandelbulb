
/**
 * @brief - Reimplementation of a program that have seen a lot
 *          of trials along the years with the notable examples
 *          listed below:
 *
 *              +-------------------------+---------------+
 *              |      Name               | Creation date |
 *              +-------------------------+---------------+
 *              |       MANDELBULB        |   11/07/2013  |
 *              +-------------------------+---------------+
 *              |    MANDELBULB VIEWER    |   02/08/2013  |
 *              +-------------------------+---------------+
 *              | MANDELBULB VIEWER REAL  |   19/06/2014  |
 *              +-------------------------+---------------+
 *              | MANDELBULB RAYTRACING   |  ~15/04/2017  |
 *              +-------------------------+---------------+
 *
 *          The goal of the tool is to provide a way to display
 *          the `Mandelbulb` which is analogous to the fractal
 *          `Mandelbrot` in 2D.
 *          Several approaches have been tried out including the
 *          use of raytracing. The approach followed here will
 *          be raytracing as well, with some sort of a deferred
 *          rendering either on the host or on a GPU device.
 *          
 *          We have to acknowledge the value of the information
 *          found in the following resource:
 *          http://celarek.at/wp/wp-content/uploads/2014/05/realTimeFractalsReport.pdf
 *          which proved very useful to implement the distance
 *          estimation techniques.
 *
 *          Implemented from:
 *            - 17/01/2020 - ??/??/2020
 */

# include <core_utils/StdLogger.hh>
# include <core_utils/LoggerLocator.hh>
# include <sdl_app_core/SdlApplication.hh>
# include <core_utils/CoreException.hh>

# include "MandelbulbRenderer.hh"
# include "InfoPanel.hh"
# include "LightSettings.hh"
# include "RenderMenu.hh"
# include "RenderSettings.hh"

int main(int /*argc*/, char** /*argv*/) {
  // Create the logger.
  utils::StdLogger logger;
  utils::LoggerLocator::provide(&logger);

  const std::string service("mandelbulb");
  const std::string module("main");

  // Create the application window parameters.
  const std::string appName = std::string("mandelbulb");
  const std::string appTitle = std::string("I saw an interior designer running away in fear earlier");
  const std::string appIcon = std::string("data/img/brute.bmp");
  const utils::Sizei size(640, 480);

  const float eventsFPS = 60.0f;
  const float renderFPS = 50.0f;

  sdl::app::SdlApplicationShPtr app = nullptr;

  try {
    app = std::make_shared<sdl::app::SdlApplication>(
      appName,
      appTitle,
      appIcon,
      size,
      true,
      utils::Sizef(0.7f, 0.5f),
      renderFPS,
      eventsFPS
    );

    // Create the layout of the window: the main tab is a scrollable widget
    // allowing the display of the mandelbulb. The rigth dock widget allows
    // to control the computation parameters of this object.
    const float fov = 40.0f;
    std::vector<float> transform(16u, 0.0f);
    const float distance = 2.0f;

    mandelbulb::CameraShPtr cam = std::make_shared<mandelbulb::Camera>(
      utils::Sizei(512, 256),
      fov,
      distance,
      utils::Vector2f()
    );

    mandelbulb::RenderProperties props{10u, 8.0f, 0.01f, 50u, 4.0f};
    mandelbulb::FractalShPtr fractal = std::make_shared<mandelbulb::Fractal>(cam, props);

    mandelbulb::RenderMenu* menu = new mandelbulb::RenderMenu();
    app->setMenuBar(menu);

    mandelbulb::MandelbulbRenderer* renderer = new mandelbulb::MandelbulbRenderer(fractal);
    app->setCentralWidget(renderer);

    mandelbulb::RenderSettings* render = new mandelbulb::RenderSettings();
    app->addDockWidget(render, sdl::app::DockWidgetArea::RightArea, "Render");

    mandelbulb::LightSettings* lights = new mandelbulb::LightSettings();
    app->addDockWidget(lights, sdl::app::DockWidgetArea::RightArea, "Lights");

    mandelbulb::InfoPanel* info = new mandelbulb::InfoPanel();
    app->setStatusBar(info);

    // Connect signals and slots of various components of the `UI`.
    int slot1 = renderer->onCoordinatesChanged.connect_member<mandelbulb::InfoPanel>(
      info,
      &mandelbulb::InfoPanel::onCoordinatesChanged
    );
    int slot2 = renderer->onDepthChanged.connect_member<mandelbulb::InfoPanel>(
      info,
      &mandelbulb::InfoPanel::onDepthChanged
    );

    int slot3 = fractal->onRenderingCompletionAdvanced.connect_member<mandelbulb::RenderMenu>(
      menu,
      &mandelbulb::RenderMenu::onCompletionChanged
    );

    int slot4 = render->onRenderingSettingsChanged.connect_member<mandelbulb::Fractal>(
      fractal.get(),
      &mandelbulb::Fractal::onRenderingPropsChanged
    );

    // Run it.
    app->run();

    // Disconnect signals.
    renderer->onCoordinatesChanged.disconnect(slot1);
    renderer->onDepthChanged.disconnect(slot2);

    fractal->onRenderingCompletionAdvanced.disconnect(slot3);
    render->onRenderingSettingsChanged.disconnect(slot4);
  }
  catch (const utils::CoreException& e) {
    utils::LoggerLocator::getLogger().logMessage(
      utils::Level::Critical,
      std::string("Caught internal exception while setting up application"),
      module,
      service,
      e.what()
    );
  }
  catch (const std::exception& e) {
    utils::LoggerLocator::getLogger().logMessage(
      utils::Level::Critical,
      std::string("Caught exception while setting up application"),
      module,
      service,
      e.what()
    );
  }
  catch (...) {
    utils::LoggerLocator::getLogger().logMessage(
      utils::Level::Critical,
      std::string("Unexpected error while setting up application"),
      module,
      service
    );
  }

  app.reset();

  // All is good.
  return EXIT_SUCCESS;
}
