
/**
 * @brief - Reimplementation of a program that have seen a lot of trials along
 *          the years with the notable examples listed below:
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
 *          The goal of the tool is to provide a way to display the `Mandelbulb`
 *          which is analogous to the `Mandelbrot` set fractal in 2D.
 *          Several approaches have been tried out including the use of raytracing
 *          (even though it does not seem to have paid off). The approach followed
 *          here will be raytracing as well, with some sort of a deferred rendering
 *          on a GPU device.
 *          We have to acknowledge the value of the information found in the link
 *          below:
 *          http://celarek.at/wp/wp-content/uploads/2014/05/realTimeFractalsReport.pdf
 *          which proved very useful to implement the distance estimation techniques.
 *
 *          Implemented from:
 *            - 17/01/2020 - 29/01/2020
 */

# include <core_utils/log/StdLogger.hh>
# include <core_utils/log/PrefixedLogger.hh>
# include <core_utils/log/Locator.hh>
# include <sdl_app_core/SdlApplication.hh>
# include <core_utils/CoreException.hh>

# include "MandelbulbRenderer.hh"
# include "InfoPanel.hh"
# include "LightSettings.hh"
# include "RenderMenu.hh"
# include "RenderSettings.hh"
# include "FractalSettings.hh"

namespace {
constexpr auto APP_NAME = "mandelbulb";
constexpr auto APP_TITLE = "I saw an interior designer running away in fear earlier";
constexpr auto APP_ICON_PATH = "data/img/brute.bmp";
}

int main(int /*argc*/, char** /*argv*/) {
  // Create the logger.
  utils::log::StdLogger raw;
  raw.setLevel(utils::log::Severity::DEBUG);
  utils::log::PrefixedLogger logger("mandelbulb", "main");
  utils::log::Locator::provide(&raw);

  const std::string service("mandelbulb");
  const std::string module("main");

  const float eventsFPS = 60.0f;
  const float renderFPS = 50.0f;

  sdl::app::SdlApplicationShPtr app = nullptr;

  try {
    app = std::make_shared<sdl::app::SdlApplication>(
      APP_NAME,
      APP_TITLE,
      APP_ICON_PATH,
      utils::Sizei(640, 480),
      true,
      utils::Sizef(0.7f, 0.5f),
      renderFPS,
      eventsFPS
    );

    // Create the layout of the window: the main tab is a scrollable widget
    // allowing the display of the mandelbulb. The right dock widget allows
    // to control the computation parameters of this object.
    const float fov = 40.0f;
    const float distance = 6.83f;

    mandelbulb::CameraShPtr cam = std::make_shared<mandelbulb::Camera>(
      utils::Sizei(512, 256),
      fov,
      distance,
      utils::Vector2f()
    );

    mandelbulb::RenderProperties rProps{10u, 8.0f, 0.001f, 100u, 8.0f};
    mandelbulb::ShadingProperties sProps{
      sdl::core::engine::Color::NamedColor::Maroon,
      0.7f,
      sdl::core::engine::Color::NamedColor::Black,

      1.0f,
      0.1f
    };
    mandelbulb::FractalShPtr fractal = std::make_shared<mandelbulb::Fractal>(
      cam,
      rProps,
      sProps,
      mandelbulb::LightSettings::generateDefaultLights()
    );

    mandelbulb::RenderMenu* menu = new mandelbulb::RenderMenu();
    app->setMenuBar(menu);

    mandelbulb::MandelbulbRenderer* renderer = new mandelbulb::MandelbulbRenderer(fractal);
    app->setCentralWidget(renderer);

    mandelbulb::RenderSettings* render = new mandelbulb::RenderSettings();
    app->addDockWidget(render, sdl::app::DockWidgetArea::RightArea, "Render");

    mandelbulb::LightSettings* lights = new mandelbulb::LightSettings();
    app->addDockWidget(lights, sdl::app::DockWidgetArea::RightArea, "Lights");

    mandelbulb::FractalSettings* settings = new mandelbulb::FractalSettings();
    app->addDockWidget(settings, sdl::app::DockWidgetArea::RightArea, "Fractal");

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

    int slot5 = lights->onLightsChanged.connect_member<mandelbulb::Fractal>(
      fractal.get(),
      &mandelbulb::Fractal::onLightsChanged
    );

    int slot6 = settings->onShadingPropertiesChanged.connect_member<mandelbulb::Fractal>(
      fractal.get(),
      &mandelbulb::Fractal::onShadingPropsChanged
    );

    // Run it.
    app->run();

    // Disconnect signals.
    renderer->onCoordinatesChanged.disconnect(slot1);
    renderer->onDepthChanged.disconnect(slot2);

    fractal->onRenderingCompletionAdvanced.disconnect(slot3);
    render->onRenderingSettingsChanged.disconnect(slot4);

    lights->onLightsChanged.disconnect(slot5);

    settings->onShadingPropertiesChanged.disconnect(slot6);

    app.reset();
  }
  catch (const utils::CoreException& e) {
    logger.error("Caught internal exception while setting up application", e.what());
    return EXIT_FAILURE;
  }
  catch (const std::exception& e) {
    logger.error("Caught internal exception while setting up application", e.what());
    return EXIT_FAILURE;
  }
  catch (...) {
    logger.error("Unexpected error while setting up application");
    return EXIT_FAILURE;
  }

  // All is good.
  return EXIT_SUCCESS;
}
