#ifndef    LIGHT_HXX
# define   LIGHT_HXX

# include "Light.hh"

namespace mandelbulb {

  inline
  Light::Light(const utils::Vector3f& dir):
    utils::CoreObject(std::string("light_") + dir.toString()),

    m_direction(dir),
    m_color(getDefaultColor()),
    m_intensity(getDefaultIntensity())
  {
    setService("light");
  }

  inline
  utils::Vector3f
  Light::getDirection() const noexcept {
    return m_direction;
  }

  inline
  sdl::core::engine::Color
  Light::getColor() const noexcept {
    return m_color;
  }

  inline
  void
  Light::setColor(const sdl::core::engine::Color& color) noexcept {
    m_color = color;
  }

  inline
  float
  Light::getIntensity() const noexcept {
    return m_intensity;
  }

  inline
  void
  Light::setIntensity(float intensity) {
    if (intensity < 0.0f) {
      log(
        std::string("Cannot set intensity for light to ") + std::to_string(intensity) +
        " keeping current value of " + std::to_string(getIntensity()),
        utils::Level::Warning
      );

      return;
    }

    m_intensity = intensity;
  }

  inline
  sdl::core::engine::Color
  Light::getDefaultColor() noexcept {
    return sdl::core::engine::Color::NamedColor::White;
  }

  inline
  float
  Light::getDefaultIntensity() noexcept {
    return 1.0f;
  }

}

#endif    /* LIGHT_HXX */
