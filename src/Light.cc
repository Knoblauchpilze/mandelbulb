
# include "Light.hh"

namespace mandelbulb {

  LightShPtr
  Light::fromPositionAndTarget(const utils::Vector3f& pos,
                               const utils::Vector3f& target) noexcept
  {
    // Convert the position and target to a direction.
    utils::Vector3f dir = (target - pos);

    // Normalize it and use it to create a light.
    return std::make_shared<Light>(dir.normalized());
  }

}
