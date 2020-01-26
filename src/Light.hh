#ifndef    LIGHT_HH
# define   LIGHT_HH

# include <memory>
# include <core_utils/CoreObject.hh>
# include <maths_utils/Vector3.hh>
# include <sdl_engine/Color.hh>

namespace mandelbulb {

  /// Forward declaration to be able to use a `LightShPtr` right away.
  class Light;
  using LightShPtr = std::shared_ptr<Light>;

  class Light: public utils::CoreObject {
    public:

      Light(const utils::Vector3f& dir);

      virtual ~Light() = default;

      static
      LightShPtr
      fromPositionAndTarget(const utils::Vector3f& pos,
                            const utils::Vector3f& target) noexcept;

      utils::Vector3f
      getDirection() const noexcept;

      sdl::core::engine::Color
      getColor() const noexcept;

      void
      setColor(const sdl::core::engine::Color& color) noexcept;

      float
      getIntensity() const noexcept;

      void
      setIntensity(float intensity);

    private:

      sdl::core::engine::Color
      getDefaultColor() noexcept;

      float
      getDefaultIntensity() noexcept;

    private:

      utils::Vector3f m_direction;

      sdl::core::engine::Color m_color;

      float m_intensity;
  };

}

# include "Light.hxx"

#endif    /* LIGHT_HH */
